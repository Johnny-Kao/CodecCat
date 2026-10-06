from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from contextlib import contextmanager

import joblib
import numpy as np

import charset_external_scorer_domain_shift_ab as base
import charset_external_scorer_learning_curve as lc
import charset_candidate_calibration_ab as calmod
import r12_rl_candidate_specialist as r12
import r13_representation_sufficiency as r13
import r16a_fused_full_b as r16
import r17b_ambiguity_gated_full_b as r17

OUTER_FOLDS=4
C_VALUES=(0.40,0.50,0.60)
CENTER=0.50
R12_THRESHOLD=0.65
GATE_FRACTION=0.50

ORIG_FIT=lc.fit_route_models


def fit_route_models_c(c):
    def _fit(ext_rows, legacy_X, legacy_y, legacy_b):
        models={}
        for route in ("U","N","RL","RH"):
            li=np.where(legacy_b==route)[0]
            Xtr=legacy_X[li]
            ytr=legacy_y[li]

            er=[s for s in ext_rows if s["route"]==route]
            if er:
                Xext=np.stack([base.scorer_features(s["data"]) for s in er])
                yext=np.array([s["label"] for s in er],dtype=object)
                known=np.isin(yext,np.unique(legacy_y))
                if known.any():
                    Xtr=np.concatenate([Xtr,Xext[known]],axis=0)
                    ytr=np.concatenate([ytr,yext[known]],axis=0)

            if len(Xtr)>=20 and len(np.unique(ytr))>=2:
                models[route]=base.fit_linear(Xtr,ytr,C=c)
        return models
    return _fit


@contextmanager
def route_c(c):
    lc.fit_route_models=fit_route_models_c(c)
    try:
        yield
    finally:
        lc.fit_route_models=ORIG_FIT


def train_thresholds(stack,train_pool):
    by_route={"U":[],"RH":[]}

    def _collect():
        for s in train_pool:
            if s["route"] not in by_route:
                continue
            model=stack["models"].get(s["route"])
            if model is None:
                continue
            _,scores=calmod.raw_rank_scores(model,s["data"])
            if len(scores)>=2:
                by_route[s["route"]].append(float(scores[0]-scores[1]))

    r17.with_scorer(stack["scorer"],_collect)

    out={}
    for route,vals in by_route.items():
        a=np.asarray(vals,dtype=np.float64)
        out[route]=float(np.quantile(a,GATE_FRACTION)) if len(a) else float("-inf")
    return out,{k:len(v) for k,v in by_route.items()}


def add(c,route,hit):
    c["n"]+=1
    c["hits"]+=hit
    c[f"{route}_n"]+=1
    c[f"{route}_hits"]+=hit


def pack(c):
    return {
        "n":c["n"],
        "hits":c["hits"],
        "top1":c["hits"]/max(1,c["n"]),
        "eligible":c["eligible"],
        "escalated":c["escalated"],
        "escalation_rate_all":c["escalated"]/max(1,c["n"]),
        "escalation_rate_eligible":c["escalated"]/max(1,c["eligible"]),
        "beneficial":c["beneficial"],
        "harmful":c["harmful"],
        "routes":{
            r:{
                "n":c[f"{r}_n"],
                "hits":c[f"{r}_hits"],
                "top1":c[f"{r}_hits"]/max(1,c[f"{r}_n"])
            }
            for r in ("U","N","RL","RH") if c[f"{r}_n"]
        }
    }


def eval_c(c,external,fold_assign,base_X,full_X,legacy_y,legacy_b,full_y,full_b):
    total=Counter()
    folds=[]

    for fold in range(OUTER_FOLDS):
        train_pool=[s for s in external if fold_assign[s["warc_path"]]!=fold]
        test_rows=[s for s in external if fold_assign[s["warc_path"]]==fold]

        with route_c(c):
            bstack=r17.fit_stack(r13.baseline_features,train_pool,base_X,legacy_y,legacy_b,True)
            fstack=r17.fit_stack(r16.fused_full_b,train_pool,full_X,full_y,full_b,False)

        thresholds,margin_n=train_thresholds(bstack,train_pool)
        fc=Counter()

        for s in test_rows:
            bout,bmargin=r17.predict(bstack,s,True)
            if bout is None:
                continue
            truth=s["label"]
            bhit=int(bout[0]==truth)

            fout=None
            if s["route"] in ("U","RH"):
                fout,_=r17.predict(fstack,s,False)

            eligible=s["route"] in ("U","RH")
            if eligible:
                total["eligible"]+=1
                fc["eligible"]+=1

            escalate=(
                eligible
                and bmargin is not None
                and bmargin<=thresholds[s["route"]]
            )

            out=(fout if fout is not None else bout) if escalate else bout
            hit=int(out and out[0]==truth)
            add(total,s["route"],hit)
            add(fc,s["route"],hit)

            if escalate:
                total["escalated"]+=1
                fc["escalated"]+=1
                if hit>bhit:
                    total["beneficial"]+=1
                    fc["beneficial"]+=1
                elif hit<bhit:
                    total["harmful"]+=1
                    fc["harmful"]+=1

        folds.append({
            "fold":fold,
            "result":pack(fc),
            "thresholds":thresholds,
            "margin_training_n":margin_n,
        })

    return {"pooled":pack(total),"folds":folds}


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
        p:int.from_bytes(
            hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big"
        )%OUTER_FOLDS
        for p in paths
    }

    results={str(c):eval_c(
        c,external,fold_assign,base_X,full_X,legacy_y,legacy_b,full_y,full_b
    ) for c in C_VALUES}

    hits=[results[str(c)]["pooled"]["hits"] for c in C_VALUES]
    center_hits=results[str(CENTER)]["pooled"]["hits"]
    escalation_rates=[
        results[str(c)]["pooled"]["escalation_rate_all"] for c in C_VALUES
    ]

    print(json.dumps({
        "phase":"r19c_system_c_suitability",
        "purpose":"Validate C=0.5 at the final cascade level; this is a robustness audit, not parameter selection.",
        "fixed_policy":{
            "r12_threshold":R12_THRESHOLD,
            "gate_fraction":GATE_FRACTION,
            "gate_definition":"training-side per-route raw margin quantile",
            "full_b_routes":["U","RH"],
        },
        "values":C_VALUES,
        "center":CENTER,
        "results":results,
        "plateau":{
            "center_hits":center_hits,
            "min_hits":min(hits),
            "max_hits":max(hits),
            "range_hits":max(hits)-min(hits),
            "center_within_one_hit_of_best":max(hits)-center_hits<=1,
            "accuracy_plateau_within_two_hits":max(hits)-min(hits)<=2,
            "escalation_rate_range_pp":100.0*(max(escalation_rates)-min(escalation_rates)),
        },
        "acceptance_rule":"Treat C=0.5 as system-level suitable only if center is within one hit of local best, total accuracy range is <=2 hits, and no route/fold exhibits a sharp collapse. Escalation-rate variation is reported descriptively, not optimized.",
        "methodology":{
            "one_axis_only":True,
            "all_non_C_policy_locked":True,
            "same_outer_folds":True,
            "same_training_corpus":True,
            "no_new_features":True,
            "no_new_model_family":True,
            "cc_main_2026_30_used":False,
        }
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
