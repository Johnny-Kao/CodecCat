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
import r12_rl_candidate_specialist as r12
import r13_representation_sufficiency as r13
import r16a_fused_full_b as r16
import r17b_ambiguity_gated_full_b as r17
import r20a_cached_feature_path as r20a

OUTER_FOLDS=4
TRAIN_N=300
R12_THRESHOLD=0.65
GATE_FRACTION=0.50


def fit_stack_cached(scorer,train_pool,legacy_X,legacy_y,legacy_b,with_rl):
    return r17.fit_stack(scorer,train_pool,legacy_X,legacy_y,legacy_b,with_rl)


def train_thresholds(stack,train_pool):
    vals={"U":[],"RH":[]}
    def _collect():
        for s in train_pool:
            if s["route"] not in vals:
                continue
            model=stack["models"].get(s["route"])
            if model is None:
                continue
            _,scores=calmod.raw_rank_scores(model,s["data"])
            if len(scores)>=2:
                vals[s["route"]].append(float(scores[0]-scores[1]))
    r17.with_scorer(stack["scorer"],_collect)
    return {
        r:float(np.quantile(np.asarray(v,dtype=np.float64),GATE_FRACTION)) if v else float("-inf")
        for r,v in vals.items()
    }


def predict_with_cached_features(bstack,fstack,s,thresholds):
    route=s["route"]
    a=r20a.hmt_array(s["data"])
    bfeat=r20a.baseline_from_array(a)

    model=bstack["models"].get(route)
    if model is None:
        return None,False

    def _baseline_predict():
        old=base.scorer_features
        base.scorer_features=lambda _: bfeat
        try:
            rank,scores=calmod.raw_rank_scores(model,s["data"])
            margin=float(scores[0]-scores[1]) if len(scores)>=2 else float("inf")
            _,hybrid=tri.make_hybrid(model,bstack["cal"],s,bstack["classes"],bstack["families"])
            gated=gate050.apply_gated(bstack["triad"],model,s,hybrid)
            sig=canon.apply_one(bstack["sig"],canon.SIG_PAIR,model,s,gated)
            out=canon.apply_guarded_gb(bstack["gb"],model,s,sig)
            if bstack["spec"] is not None:
                out,_=r12.apply_specialist(bstack["spec"],model,s,out,R12_THRESHOLD)
            return out,margin
        finally:
            base.scorer_features=old

    bout,margin=_baseline_predict()
    if bout is None:
        return None,False

    escalate=(
        route in ("U","RH")
        and margin<=thresholds[route]
    )
    if not escalate:
        return bout,False

    efeat=r20a.extra_from_array(a,route)
    ffeat=np.concatenate([bfeat,efeat])
    fmodel=fstack["models"].get(route)
    if fmodel is None:
        return bout,False

    old=base.scorer_features
    base.scorer_features=lambda _: ffeat
    try:
        _,hybrid=tri.make_hybrid(fmodel,fstack["cal"],s,fstack["classes"],fstack["families"])
        gated=gate050.apply_gated(fstack["triad"],fmodel,s,hybrid)
        sig=canon.apply_one(fstack["sig"],canon.SIG_PAIR,fmodel,s,gated)
        fout=canon.apply_guarded_gb(fstack["gb"],fmodel,s,sig)
    finally:
        base.scorer_features=old
    return fout,True


def predict_reference(bstack,fstack,s,thresholds):
    bout,margin=r17.predict(bstack,s,True)
    if bout is None:
        return None,False
    escalate=(
        s["route"] in ("U","RH")
        and margin is not None
        and margin<=thresholds[s["route"]]
    )
    if not escalate:
        return bout,False
    fout,_=r17.predict(fstack,s,False)
    return (fout if fout is not None else bout),True


def timed(rows,fn,repeats=7):
    vals=[]
    for r in rows:
        fn(r)
    per_sample_all=[]
    for _ in range(repeats):
        sample_times=[]
        for r in rows:
            t=time.perf_counter_ns()
            fn(r)
            sample_times.append((time.perf_counter_ns()-t)/1000.0)
        vals.append(float(np.median(sample_times)))
        per_sample_all.extend(sample_times)
    a=np.asarray(per_sample_all,dtype=np.float64)
    return {
        "median_us":float(np.median(a)),
        "p90_us":float(np.quantile(a,.90)),
        "p95_us":float(np.quantile(a,.95)),
        "p99_us":float(np.quantile(a,.99)),
        "round_median_of_medians_us":float(np.median(vals)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",required=True)
    args=ap.parse_args()

    state=joblib.load(args.state)
    external=r17.reconstruct_external(state)
    legacy_rows=r17.raw_legacy_rows()

    base_X,legacy_y,legacy_b=r17.legacy_arrays(r13.baseline_features,legacy_rows)
    full_X,full_y,full_b=r17.legacy_arrays(r16.fused_full_b,legacy_rows)

    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={
        p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS
        for p in paths
    }

    pooled=Counter()
    folds=[]
    timing_ref=[]
    timing_cached=[]

    for fold in range(OUTER_FOLDS):
        train_pool=[s for s in external if fold_assign[s["warc_path"]]!=fold]
        test_rows=[s for s in external if fold_assign[s["warc_path"]]==fold]

        bstack=fit_stack_cached(r13.baseline_features,train_pool,base_X,legacy_y,legacy_b,True)
        fstack=fit_stack_cached(r16.fused_full_b,train_pool,full_X,full_y,full_b,False)
        thresholds=train_thresholds(bstack,train_pool)

        fc=Counter()
        mismatches=0
        escalation_mismatch=0

        for s in test_rows:
            ref,refe=predict_reference(bstack,fstack,s,thresholds)
            cached,cachede=predict_with_cached_features(bstack,fstack,s,thresholds)
            if ref is None or cached is None:
                continue
            mismatches+=int(ref[0]!=cached[0])
            escalation_mismatch+=int(refe!=cachede)
            hit=int(cached[0]==s["label"])
            pooled["n"]+=1
            pooled["hits"]+=hit
            pooled["escalated"]+=int(cachede)
            fc["n"]+=1
            fc["hits"]+=hit
            fc["escalated"]+=int(cachede)

        timing_ref.append(timed(test_rows,lambda s:predict_reference(bstack,fstack,s,thresholds),repeats=5))
        timing_cached.append(timed(test_rows,lambda s:predict_with_cached_features(bstack,fstack,s,thresholds),repeats=5))

        folds.append({
            "fold":fold,
            "n":fc["n"],
            "hits":fc["hits"],
            "top1":fc["hits"]/max(1,fc["n"]),
            "escalated":fc["escalated"],
            "prediction_mismatch":mismatches,
            "escalation_mismatch":escalation_mismatch,
            "thresholds":thresholds,
        })

    def agg(ts,key):
        return float(np.median([x[key] for x in ts]))

    print(json.dumps({
        "phase":"r20b_end_to_end_cached_cascade",
        "accuracy":{
            "n":pooled["n"],
            "hits":pooled["hits"],
            "top1":pooled["hits"]/max(1,pooled["n"]),
            "escalated":pooled["escalated"],
            "escalation_rate_all":pooled["escalated"]/max(1,pooled["n"]),
        },
        "folds":folds,
        "timing_us_per_sample_median_across_folds":{
            "reference":{
                k:agg(timing_ref,k)
                for k in ("median_us","p90_us","p95_us","p99_us","round_median_of_medians_us")
            },
            "cached":{
                k:agg(timing_cached,k)
                for k in ("median_us","p90_us","p95_us","p99_us","round_median_of_medians_us")
            },
        },
        "timing_by_fold":{"reference":timing_ref,"cached":timing_cached},
        "acceptance_rule":"Accept only if all folds have zero prediction/escalation mismatches versus the frozen reference path and pooled accuracy remains 378/418. Timing comparisons are same-run only.",
        "methodology":{
            "same_fold_models":True,
            "route_once":True,
            "hmt_once":True,
            "baseline_feature_once":True,
            "conditional_extra_from_cached_hmt":True,
            "r12_threshold":R12_THRESHOLD,
            "gate_fraction":GATE_FRACTION,
            "cc_main_2026_30_used":False
        }
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
