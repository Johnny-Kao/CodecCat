from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter

import joblib
import numpy as np

import charset_external_scorer_domain_shift_ab as base
import charset_external_scorer_learning_curve as lc
import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_post_gate_050_error_decomposition as gate050
import charset_canonical_guarded_final_validation as canon
from charset_cost_aware_routing_tree import collect as collect_legacy
import r13_representation_sufficiency as r13
import r12_rl_candidate_specialist as r12

OUTER_FOLDS=4
TRAIN_N=300
R12_THRESHOLD=0.65
EXTRA_WIDTH=368


def reconstruct_external(state):
    rows=[]; seen=set()
    for fd in state["folds"]:
        for s in fd["rows"]:
            key=(s["warc_path"],s.get("host",""),s["route"],s["label"],hashlib.sha256(s["data"]).digest())
            if key not in seen:
                seen.add(key); rows.append(s)
    return rows


def raw_legacy_rows():
    rows,_=collect_legacy(); out=[]
    for row in rows:
        data=row["path"].read_bytes()
        label=base.normalize_label(base.canonical_encoding(row["encoding"])) or row["encoding"]
        out.append({"data":data,"label":label,"route":base.route_bucket(data)})
    return out


def reference_full_b(data: bytes) -> np.ndarray:
    b=r13.baseline_features(data)
    if base.route_bucket(data) not in ("U","RH"):
        return np.concatenate([b,np.zeros(EXTRA_WIDTH,dtype=np.float32)])
    return np.concatenate([b,r13.generic_order_features(data)])


def fused_full_b(data: bytes) -> np.ndarray:
    sample=r13.hmt768(data)
    a=np.frombuffer(sample,dtype=np.uint8)
    n=max(1,len(a))
    out=np.zeros(886,dtype=np.float32)

    counts=np.bincount(a,minlength=256)
    out[:256]=counts.astype(np.float32)/n

    if len(a)>=2:
        left=a[:-1]
        right=a[1:]

        wrapped=left+right
        out[256:512]=np.bincount(wrapped,minlength=256).astype(np.float32)/(len(a)-1)
    else:
        left=right=None

    if len(a):
        out[512:518]=(
            math.log2(n+1)/16.0,
            float(np.mean(a>=128)),
            float(np.mean(a==0)),
            float(np.mean((a>=32)&(a<=126))),
            float(np.mean(a==10)),
            float(np.mean(a==13)),
        )

    # Route-local: preserve R15 full-B exactly for U/RH, zero extras elsewhere.
    if base.route_bucket(data) not in ("U","RH"):
        return out

    if len(a)>=2:
        x=left.astype(np.int16,copy=False)
        y=right.astype(np.int16,copy=False)
        diff=((y-x)&255).astype(np.intp,copy=False)
        out[518:774]=np.bincount(diff,minlength=256).astype(np.float32)/(len(a)-1)

        xb=(np.bitwise_xor(left,right)>>2).astype(np.intp,copy=False)
        out[774:838]=np.bincount(xb,minlength=64).astype(np.float32)/(len(a)-1)

    # Position-aware high-byte distribution, reusing the already sampled buffer.
    p=838
    for chunk in (a[:256],a[256:512],a[512:768]):
        high=chunk[chunk>=128]
        if len(high):
            idx=((high.astype(np.uint16)-128)>>3).astype(np.intp,copy=False)
            out[p:p+16]=np.bincount(idx,minlength=16).astype(np.float32)/len(high)
        p+=16
    return out


def feature_cost_us(fn,samples,repeats=7):
    vals=[]
    for _ in range(repeats):
        t=time.perf_counter_ns()
        for d in samples: fn(d)
        vals.append((time.perf_counter_ns()-t)/1000/max(1,len(samples)))
    return float(np.median(vals))


def equivalence(rows):
    mismatch=0; max_abs=0.0
    for r in rows:
        a=reference_full_b(r["data"])
        b=fused_full_b(r["data"])
        diff=float(np.max(np.abs(a-b))) if len(a) else 0.0
        max_abs=max(max_abs,diff)
        if not np.array_equal(a,b):
            mismatch+=1
    return mismatch,max_abs


def evaluate(external,fold_assign,legacy_rows):
    old=base.scorer_features; base.scorer_features=fused_full_b
    try:
        legacy_X=np.stack([fused_full_b(r["data"]) for r in legacy_rows])
        legacy_y=np.asarray([r["label"] for r in legacy_rows],dtype=object)
        legacy_b=np.asarray([r["route"] for r in legacy_rows],dtype=object)
        classes,families=calmod.build_vocab(legacy_y)
        total=Counter(); routes={r:Counter() for r in ("U","N","RL","RH")}; folds=[]
        for fold in range(OUTER_FOLDS):
            train_pool=[s for s in external if fold_assign[s["warc_path"]]!=fold]
            test_rows=[s for s in external if fold_assign[s["warc_path"]]==fold]
            ext_train=lc.deterministic_nested_subset(train_pool,TRAIN_N)
            models=lc.fit_route_models(ext_train,legacy_X,legacy_y,legacy_b)
            cal=calmod.crossfit_calibration_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
            triad_cal=tri.crossfit_triad_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
            sig_cal=canon.fit_one(canon.SIG_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)
            gb_cal=canon.fit_one(canon.GB_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)
            spec=r12.fit_specialist(train_pool,legacy_X,legacy_y,legacy_b)
            fc=Counter()
            for s in test_rows:
                model=models.get(s["route"])
                if model is None or s["label"] not in model[1].classes_: continue
                _,hybrid=tri.make_hybrid(model,cal,s,classes,families)
                gated=gate050.apply_gated(triad_cal,model,s,hybrid)
                sig=canon.apply_one(sig_cal,canon.SIG_PAIR,model,s,gated)
                out=canon.apply_guarded_gb(gb_cal,model,s,sig)
                out,_=r12.apply_specialist(spec,model,s,out,R12_THRESHOLD)
                hit=int(out and out[0]==s["label"])
                total["n"]+=1; total["hits"]+=hit
                routes[s["route"]]["n"]+=1; routes[s["route"]]["hits"]+=hit
                fc["n"]+=1; fc["hits"]+=hit
            folds.append({"fold":fold,"n":fc["n"],"hits":fc["hits"],"top1":fc["hits"]/max(1,fc["n"])})
        return {
            "n":total["n"],"hits":total["hits"],"top1":total["hits"]/max(1,total["n"]),
            "routes":{r:{"n":v["n"],"hits":v["hits"],"top1":v["hits"]/max(1,v["n"])} for r,v in routes.items() if v["n"]},
            "folds":folds,
        }
    finally:
        base.scorer_features=old


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    external=reconstruct_external(state); legacy_rows=raw_legacy_rows()

    all_rows=legacy_rows+external
    mismatch,max_abs=equivalence(all_rows)

    samples=[s["data"] for s in external[:128]]
    ref_cost=feature_cost_us(reference_full_b,samples)
    fused_cost=feature_cost_us(fused_full_b,samples)
    baseline_cost=feature_cost_us(r13.baseline_features,samples)

    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}
    result=evaluate(external,fold_assign,legacy_rows)

    print(json.dumps({
        "phase":"r16a_fused_full_b",
        "equivalence":{"rows_checked":len(all_rows),"array_mismatch":mismatch,"max_abs_diff":max_abs},
        "cost_us_per_sample":{
            "baseline":baseline_cost,
            "reference_full_b":ref_cost,
            "fused_full_b":fused_cost,
            "fused_vs_baseline":fused_cost/max(1e-9,baseline_cost),
            "fused_vs_reference":fused_cost/max(1e-9,ref_cost),
        },
        "accuracy":result,
        "acceptance_rule":"Accept fused extraction only with zero feature mismatches and the same 378/417 full-B+R12 result.",
        "methodology":{"same_rows":True,"same_folds":True,"r12_threshold":0.65,"cc_main_2026_30_used":False}
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
