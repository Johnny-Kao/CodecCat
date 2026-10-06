from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import joblib
import numpy as np

import charset_external_scorer_domain_shift_ab as base
import r13_representation_sufficiency as r13
import r16a_fused_full_b as r16

EXTRA_WIDTH=368


def reconstruct_external(state):
    rows=[]; seen=set()
    for fd in state["folds"]:
        for s in fd["rows"]:
            key=(s["warc_path"],s.get("host",""),s["route"],s["label"],hashlib.sha256(s["data"]).digest())
            if key not in seen:
                seen.add(key); rows.append(s)
    return rows


def generic_extra_prerouted(data: bytes, route_name: str) -> np.ndarray:
    out=np.zeros(EXTRA_WIDTH,dtype=np.float32)
    if route_name not in ("U","RH"):
        return out

    a=np.frombuffer(r13.hmt768(data),dtype=np.uint8)
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


def full_b_prerouted(data: bytes, route_name: str) -> np.ndarray:
    return np.concatenate([r13.baseline_features(data),generic_extra_prerouted(data,route_name)])


def timed_rows(rows, fn, repeats=9):
    vals=[]
    for _ in range(repeats):
        t=time.perf_counter_ns()
        for r in rows:
            fn(r)
        vals.append((time.perf_counter_ns()-t)/1000.0/max(1,len(rows)))
    return float(np.median(vals))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    rows=reconstruct_external(state)

    mismatch=0; max_abs=0.0
    for r in rows:
        a=r16.fused_full_b(r["data"])
        b=full_b_prerouted(r["data"],r["route"])
        d=float(np.max(np.abs(a-b)))
        max_abs=max(max_abs,d)
        mismatch += int(not np.array_equal(a,b))

    base_cost=timed_rows(rows,lambda r:r13.baseline_features(r["data"]))
    extra_all=timed_rows(rows,lambda r:generic_extra_prerouted(r["data"],r["route"]))
    full_cost=timed_rows(rows,lambda r:full_b_prerouted(r["data"],r["route"]))
    route_cost=timed_rows(rows,lambda r:base.route_bucket(r["data"]))
    route_plus_base=timed_rows(rows,lambda r:(base.route_bucket(r["data"]),r13.baseline_features(r["data"])))
    route_plus_full=timed_rows(rows,lambda r:(lambda rr:full_b_prerouted(r["data"],rr))(base.route_bucket(r["data"])))

    by_route={}
    for route_name in ("U","N","RL","RH"):
        rr=[r for r in rows if r["route"]==route_name]
        if not rr: continue
        by_route[route_name]={
            "n":len(rr),
            "baseline_us":timed_rows(rr,lambda r:r13.baseline_features(r["data"]),repeats=5),
            "extra_only_us":timed_rows(rr,lambda r:generic_extra_prerouted(r["data"],r["route"]),repeats=5),
            "full_prerouted_us":timed_rows(rr,lambda r:full_b_prerouted(r["data"],r["route"]),repeats=5),
        }

    print(json.dumps({
        "phase":"r17a_prerouted_incremental_cost",
        "equivalence_to_r16a":{"rows":len(rows),"array_mismatch":mismatch,"max_abs_diff":max_abs},
        "cost_us_per_sample":{
            "route_only":route_cost,
            "baseline_features_only":base_cost,
            "generic_extra_only_weighted_all_routes":extra_all,
            "full_b_prerouted":full_cost,
            "route_plus_baseline":route_plus_base,
            "route_plus_full_b":route_plus_full,
            "full_prerouted_vs_baseline_features":full_cost/max(1e-9,base_cost),
            "route_plus_full_vs_route_plus_baseline":route_plus_full/max(1e-9,route_plus_base),
        },
        "by_route":by_route,
        "production_model":"Route once; compute baseline once; only ambiguous U/RH samples pay generic_extra_only; concatenate cached baseline+extra for full-B scoring.",
        "methodology":{"cc_main_2026_30_used":False,"no_model_refit_needed_because_bit_exact_features":True}
    },indent=2,sort_keys=True))


if __name__=="__main__": main()
