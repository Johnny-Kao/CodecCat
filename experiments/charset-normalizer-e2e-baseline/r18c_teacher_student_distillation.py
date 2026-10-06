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
TEMPERATURE=2.0
HARD_ALPHA=0.5
TOPK_SOFT=3
ORIG_FIT=lc.fit_route_models

FULL_LEGACY_X=None
FULL_LEGACY_Y=None
FULL_LEGACY_B=None


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


def softmax_temperature(scores,T):
    z=np.asarray(scores,dtype=np.float64)/T
    z=z-np.max(z,axis=1,keepdims=True)
    e=np.exp(z)
    return e/np.maximum(np.sum(e,axis=1,keepdims=True),1e-30)


def distill_fit_route_models(ext_rows,legacy_X,legacy_y,legacy_b):
    models={}
    for route in ("U","N","RL","RH"):
        li=np.where(legacy_b==route)[0]
        Xb=legacy_X[li]
        ytr=legacy_y[li]

        fi=np.where(FULL_LEGACY_B==route)[0]
        Xf=FULL_LEGACY_X[fi]
        fy=FULL_LEGACY_Y[fi]
        if not np.array_equal(ytr,fy):
            raise RuntimeError("legacy label alignment mismatch")

        er=[s for s in ext_rows if s["route"]==route]
        if er:
            yext=np.asarray([s["label"] for s in er],dtype=object)
            known=np.isin(yext,np.unique(legacy_y))
            if known.any():
                kept=[s for s,k in zip(er,known) if k]
                Xb_ext=np.stack([r13.baseline_features(s["data"]) for s in kept])
                Xf_ext=np.stack([r16.fused_full_b(s["data"]) for s in kept])
                yk=yext[known]
                Xb=np.concatenate([Xb,Xb_ext],axis=0)
                Xf=np.concatenate([Xf,Xf_ext],axis=0)
                ytr=np.concatenate([ytr,yk],axis=0)

        if len(Xb)<20 or len(np.unique(ytr))<2:
            continue

        if route not in ("U","RH"):
            models[route]=base.fit_linear(Xb,ytr,C=0.5)
            continue

        teacher=base.fit_linear(Xf,ytr,C=0.5)
        tsc,tmodel=teacher
        tscores=tmodel.decision_function(tsc.transform(Xf))
        if tscores.ndim==1:
            tscores=np.column_stack([-tscores,tscores])
        probs=softmax_temperature(tscores,TEMPERATURE)

        ssc=StandardScaler()
        Xs=ssc.fit_transform(Xb)
        Xrows=[]; yrows=[]; weights=[]
        tclasses=np.asarray(tmodel.classes_,dtype=object)

        for i in range(len(ytr)):
            Xrows.append(Xs[i]); yrows.append(ytr[i]); weights.append(HARD_ALPHA)

            order=np.argsort(probs[i])[::-1][:TOPK_SOFT]
            mass=float(np.sum(probs[i,order]))
            if mass<=0:
                continue
            for j in order:
                Xrows.append(Xs[i])
                yrows.append(tclasses[j])
                weights.append((1.0-HARD_ALPHA)*float(probs[i,j])/mass)

        Xexp=np.stack(Xrows)
        yexp=np.asarray(yrows,dtype=object)
        w=np.asarray(weights,dtype=np.float64)
        student=LogisticRegression(max_iter=2500,solver="lbfgs",C=0.5)
        student.fit(Xexp,yexp,sample_weight=w)
        models[route]=(ssc,student)
    return models


def evaluate_arm(name,scorer,fit_fn,external,fold_assign,legacy_rows):
    old_scorer=base.scorer_features
    old_fit=lc.fit_route_models
    base.scorer_features=scorer
    lc.fit_route_models=fit_fn
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
    global FULL_LEGACY_X,FULL_LEGACY_Y,FULL_LEGACY_B
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    external=reconstruct_external(state); legacy_rows=raw_legacy_rows()

    FULL_LEGACY_X=np.stack([r16.fused_full_b(r["data"]) for r in legacy_rows])
    FULL_LEGACY_Y=np.asarray([r["label"] for r in legacy_rows],dtype=object)
    FULL_LEGACY_B=np.asarray([r["route"] for r in legacy_rows],dtype=object)

    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}

    normal=evaluate_arm("baseline_normal",r13.baseline_features,ORIG_FIT,external,fold_assign,legacy_rows)
    distilled=evaluate_arm("baseline_distilled",r13.baseline_features,distill_fit_route_models,external,fold_assign,legacy_rows)
    teacher=evaluate_arm("fullB_teacher_reference",r16.fused_full_b,ORIG_FIT,external,fold_assign,legacy_rows)

    distilled["delta_hits_vs_baseline"]=distilled["hits"]-normal["hits"]
    teacher["delta_hits_vs_baseline"]=teacher["hits"]-normal["hits"]

    print(json.dumps({
        "phase":"r18c_teacher_student_distillation",
        "results":{"baseline_normal":normal,"baseline_distilled":distilled,"fullB_teacher_reference":teacher},
        "distillation":{"teacher_features":886,"student_features":518,"routes_distilled":["U","RH"],"temperature":TEMPERATURE,"hard_label_weight":HARD_ALPHA,"teacher_topk":TOPK_SOFT},
        "runtime_contract":"The distilled student uses only the original 518-d baseline representation at inference; full-B features exist only during training.",
        "acceptance_rule":"Accept only paired cross-fold student hit improvement with unchanged 518-d inference representation and stable route/fold behavior.",
        "methodology":{"distillation_propagates_into_inner_crossfit_models":True,"RL_and_N_training_unchanged":True,"r12_threshold":0.65,"cc_main_2026_30_used":False}
    },indent=2,sort_keys=True))


if __name__=="__main__": main()
