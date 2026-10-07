from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import joblib
import numpy as np

import charset_external_scorer_domain_shift_ab as base
import r16a_fused_full_b as r16
import r17b_ambiguity_gated_full_b as r17

BASE_WIDTH=518
EXTRA_WIDTH=368
GATE_FRACTION=0.50


def hmt_array(data: bytes) -> np.ndarray:
    if len(data) <= 768:
        sample=data
    else:
        m=len(data)//2
        sample=data[:256]+data[m-128:m+128]+data[-256:]
    return np.frombuffer(sample,dtype=np.uint8)


def route_once(data: bytes) -> str:
    sample=data[:4096]
    try:
        sample.decode("utf-8","strict")
        return "U"
    except UnicodeDecodeError:
        pass
    if b"\x00" in sample:
        return "N"
    a=np.frombuffer(sample,dtype=np.uint8)
    hbr=float(np.mean(a>=128)) if len(a) else 0.0
    return "RL" if hbr<=0.02 else "RH"


def baseline_from_array(a: np.ndarray) -> np.ndarray:
    n=max(1,len(a))
    out=np.zeros(BASE_WIDTH,dtype=np.float32)
    out[:256]=np.bincount(a,minlength=256).astype(np.float32)/n
    if len(a)>=2:
        wrapped=a[:-1]+a[1:]
        out[256:512]=np.bincount(wrapped,minlength=256).astype(np.float32)/(len(a)-1)
    if len(a):
        out[512:518]=(
            math.log2(n+1)/16.0,
            float(np.mean(a>=128)),
            float(np.mean(a==0)),
            float(np.mean((a>=32)&(a<=126))),
            float(np.mean(a==10)),
            float(np.mean(a==13)),
        )
    return out


def extra_from_array(a: np.ndarray, route_name: str) -> np.ndarray:
    out=np.zeros(EXTRA_WIDTH,dtype=np.float32)
    if route_name not in ("U","RH"):
        return out
    if len(a)>=2:
        left=a[:-1]
        right=a[1:]
        x=left.astype(np.int16,copy=False)
        y=right.astype(np.int16,copy=False)
        diff=((y-x)&255).astype(np.intp,copy=False)
        out[:256]=np.bincount(diff,minlength=256).astype(np.float32)/(len(a)-1)

        xb=(np.bitwise_xor(left,right)>>2).astype(np.intp,copy=False)
        out[256:320]=np.bincount(xb,minlength=64).astype(np.float32)/(len(a)-1)

    p=320
    for chunk in (a[:256],a[256:512],a[512:768]):
        high=chunk[chunk>=128]
        if len(high):
            idx=((high.astype(np.uint16)-128)>>3).astype(np.intp,copy=False)
            out[p:p+16]=np.bincount(idx,minlength=16).astype(np.float32)/len(high)
        p+=16
    return out


def full_from_cached(data: bytes, route_name: str|None=None) -> tuple[str,np.ndarray,np.ndarray]:
    route_name=route_name or route_once(data)
    a=hmt_array(data)
    b=baseline_from_array(a)
    e=extra_from_array(a,route_name)
    return route_name,b,np.concatenate([b,e])


def timed(rows, fn, repeats=11):
    vals=[]
    # one warmup
    for r in rows:
        fn(r)
    for _ in range(repeats):
        t=time.perf_counter_ns()
        for r in rows:
            fn(r)
        vals.append((time.perf_counter_ns()-t)/1000.0/max(1,len(rows)))
    vals=np.asarray(vals,dtype=np.float64)
    return {
        "median_us":float(np.median(vals)),
        "min_us":float(np.min(vals)),
        "max_us":float(np.max(vals)),
        "p90_round_us":float(np.quantile(vals,.90)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",required=True)
    args=ap.parse_args()

    state=joblib.load(args.state)
    rows=r17.reconstruct_external(state)

    mismatch_base=0
    mismatch_full=0
    route_mismatch=0
    max_base=0.0
    max_full=0.0

    for r in rows:
        rr,b,f=full_from_cached(r["data"])
        route_mismatch+=int(rr!=r["route"])

        ref=r16.fused_full_b(r["data"])
        refb=ref[:BASE_WIDTH]
        db=float(np.max(np.abs(b-refb))) if len(b) else 0.0
        df=float(np.max(np.abs(f-ref))) if len(f) else 0.0
        max_base=max(max_base,db)
        max_full=max(max_full,df)
        mismatch_base+=int(not np.array_equal(b,refb))
        mismatch_full+=int(not np.array_equal(f,ref))

    # Synthetic gate mask is derived only from the already frozen R19C center
    # empirical escalation count. This phase measures execution mechanics only,
    # not model/gate correctness. R20B will use real model margins end-to-end.
    eligible=[i for i,r in enumerate(rows) if r["route"] in ("U","RH")]
    target_escalations=249
    mask=set(eligible[:min(target_escalations,len(eligible))])

    def cached_baseline(r):
        rr=route_once(r["data"])
        a=hmt_array(r["data"])
        return rr,baseline_from_array(a)

    def cached_conditional(item):
        i,r=item
        rr=route_once(r["data"])
        a=hmt_array(r["data"])
        b=baseline_from_array(a)
        if i in mask and rr in ("U","RH"):
            e=extra_from_array(a,rr)
            return rr,np.concatenate([b,e])
        return rr,b

    indexed=list(enumerate(rows))

    overall={
        "route_only":timed(rows,lambda r:route_once(r["data"])),
        "hmt_only":timed(rows,lambda r:hmt_array(r["data"])),
        "route_plus_cached_baseline":timed(rows,cached_baseline),
        "route_plus_cached_conditional_full":timed(indexed,cached_conditional),
        "legacy_reference_full_everywhere":timed(rows,lambda r:r16.fused_full_b(r["data"])),
    }

    by_route={}
    for route in ("U","N","RL","RH"):
        rr=[r for r in rows if r["route"]==route]
        if not rr:
            continue
        by_route[route]={
            "n":len(rr),
            "route_plus_cached_baseline":timed(rr,cached_baseline,repeats=7),
            "cached_full_if_forced":timed(
                rr,
                lambda r:(lambda ro,a,b:(ro,np.concatenate([b,extra_from_array(a,ro)])))(
                    route_once(r["data"]),
                    hmt_array(r["data"]),
                    baseline_from_array(hmt_array(r["data"]))
                ),
                repeats=7,
            ),
        }

    print(json.dumps({
        "phase":"r20a_cached_feature_path",
        "equivalence":{
            "rows":len(rows),
            "route_mismatch":route_mismatch,
            "baseline_array_mismatch":mismatch_base,
            "full_array_mismatch":mismatch_full,
            "max_abs_diff_baseline":max_base,
            "max_abs_diff_full":max_full,
        },
        "timing_us_per_sample":overall,
        "by_route":by_route,
        "conditional_cost_model":{
            "eligible_rows":len(eligible),
            "synthetic_escalations_for_mechanics_only":len(mask),
            "synthetic_escalation_rate_all":len(mask)/max(1,len(rows)),
            "note":"Mask size mirrors R19C center escalation count. Membership is intentionally synthetic because R20A isolates execution mechanics; R20B uses real trained gate decisions."
        },
        "acceptance_rule":"Advance to R20B only with zero route/feature mismatches. Timing is descriptive and must not be compared across different workflow runs as an absolute A/B.",
        "methodology":{
            "route_once_per_detection":True,
            "hmt_once_per_detection":True,
            "baseline_once_per_detection":True,
            "extra_reuses_cached_hmt":True,
            "cc_main_2026_30_used":False,
        }
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
