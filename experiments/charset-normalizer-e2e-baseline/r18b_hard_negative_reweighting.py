from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import charset_external_scorer_domain_shift_ab as base
import charset_external_scorer_learning_curve as lc
import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_post_gate_050_error_decomposition as gate050
import charset_canonical_guarded_final_validation as canon
from charset_cost_aware_routing_tree import collect as collect_legacy
import r13_representation_sufficiency as r13
import r12_rl_candidate_specialist as r12
import r16a_fused_full_b as r16

OUTER_FOLDS=4
TRAIN_N=300
R12_THRESHOLD=0.65
ALPHA=3.0
TAU=1.0
ORIG_FIT=lc.fit_route_models


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


def hard_fit_route_models(ext_rows,legacy_X,legacy_y,legacy_b):
    models={}
    for route in ("U","N","RL","RH"):
        li=np.where(legacy_b==route)[0]
        Xtr=legacy_X[li]
        ytr=legacy_y[li]

        er=[s for s in ext_rows if s["route"]==route]
        if er:
            Xext=np.stack([base.scorer_features(s["data"]) for s in er])
            yext=np.asarray([s["label"] for s in er],dtype=object)
            known=np.isin(yext,np.unique(legacy_y))
            if known.any():
                Xtr=np.concatenate([Xtr,Xext[known]],axis=0)
                ytr=np.concatenate([ytr,yext[known]],axis=0)

        if len(Xtr)<20 or len(np.unique(ytr))<2:
            continue

        scaler=StandardScaler()
        Xs=scaler.fit_transform(Xtr)
        init=LogisticRegression(max_iter=2500,solver="lbfgs",C=0.5)
        init.fit(Xs,ytr)
        scores=init.decision_function(Xs)
        if scores.ndim==1:
            scores=np.column_stack([-scores,scores])
        ci={str(c):i for i,c in enumerate(init.classes_)}
        margins=np.empty(len(ytr),dtype=np.float64)
        for i,label in enumerate(ytr):
            ti=ci[str(label)]
            true=float(scores[i,ti])
            other=np.delete(scores[i],ti)
            margins[i]=true-float(np.max(other))
        difficulty=1.0/(1.0+np.exp(np.clip(margins/TAU,-8.0,8.0)))
        weights=1.0+ALPHA*difficulty

        final=LogisticRegression(max_iter=2500,solver="lbfgs",C=0.5)
        final.fit(Xs,ytr,sample_weight=weights)
        models[route]=(scaler,final)
    return models


def evaluate_arm(name,scorer,use_hard,external,fold_assign,legacy_rows):
    old_scorer=base.scorer_features
    old_fit=lc.fit_route_models
    base.scorer_features=scorer
    lc.fit_route_models=hard_fit_route_models if use_hard else ORIG_FIT
    try:
        legacy_X=np.stack([scorer(r["data"]) for r in legacy_rows])
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
                if model is None or s["label"] not in model[1].classes_:
                    continue
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
            "name":name,"n":total["n"],"hits":total["hits"],"top1":total["hits"]/max(1,total["n"]),
            "routes":{r:{"n":v["n"],"hits":v["hits"],"top1":v["hits"]/max(1,v["n"])} for r,v in routes.items() if v["n"]},
            "folds":folds,
        }
    finally:
        base.scorer_features=old_scorer
        lc.fit_route_models=old_fit


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    external=reconstruct_external(state); legacy_rows=raw_legacy_rows()
    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}

    arms=[
        ("baseline_normal",r13.baseline_features,False),
        ("baseline_hard",r13.baseline_features,True),
        ("fullB_normal",r16.fused_full_b,False),
        ("fullB_hard",r16.fused_full_b,True),
    ]
    results={name:evaluate_arm(name,scorer,hard,external,fold_assign,legacy_rows) for name,scorer,hard in arms}
    b=results["baseline_normal"]; f=results["fullB_normal"]
    for r in results.values():
        r["delta_hits_vs_baseline_normal"]=r["hits"]-b["hits"]
        r["delta_hits_vs_same_rep_normal"]=r["hits"]-(b["hits"] if r["name"].startswith("baseline") else f["hits"])

    print(json.dumps({
        "phase":"r18b_hard_negative_reweighting",
        "results":results,
        "training_rule":"Two-pass linear training. First fit computes each sample's true-vs-strongest-wrong margin; second fit uses weight = 1 + 3*sigmoid(-margin/1.0). No inference feature/model-family change.",
        "acceptance_rule":"Accept only paired cross-fold hit improvement with no added inference computation; inspect fold and route stability.",
        "methodology":{"full_downstream_refit_per_arm":True,"hard_weighting_propagates_into_inner_crossfit_models":True,"r12_threshold":0.65,"cc_main_2026_30_used":False}
    },indent=2,sort_keys=True))


if __name__=="__main__": main()
