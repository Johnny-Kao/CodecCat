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

OUTER_FOLDS = 4
INNER_FOLDS = 2
TRAIN_N = 300
TARGETS = ("utf-8", "big5", "euc-kr", "cp1251", "iso-8859-1")
OTHER = "__other__"
THRESHOLDS = (0.45, 0.55, 0.65)


def reconstruct_external(state):
    rows=[]; seen=set()
    for fd in state["folds"]:
        for s in fd["rows"]:
            key=(s["warc_path"],s.get("host",""),s["route"],s["label"],hashlib.sha256(s["data"]).digest())
            if key not in seen:
                seen.add(key); rows.append(s)
    return rows


def raw_legacy_rows():
    rows,_=collect_legacy()
    out=[]
    for row in rows:
        data=row["path"].read_bytes()
        label=base.normalize_label(base.canonical_encoding(row["encoding"])) or row["encoding"]
        out.append({"data":data,"label":label,"route":base.route_bucket(data)})
    return out


def inner_assign(path):
    return int.from_bytes(hashlib.sha256(("r12:"+path).encode()).digest()[:8],"big") % INNER_FOLDS


def score_map(model_tuple,data):
    rank,scores=calmod.raw_rank_scores(model_tuple,data)
    return {str(c):float(s) for c,s in zip(rank,scores)}, list(rank)


def decode_metrics(data, enc):
    sample=data[:4096]
    try:
        text=sample.decode(enc,"strict")
        valid=1.0
        repl=0.0
    except UnicodeDecodeError:
        valid=0.0
        text=sample.decode(enc,"replace")
        repl=text.count("\ufffd")/max(1,len(text))
    return valid,repl


def specialist_feature(model_tuple,s):
    smap,rank=score_map(model_tuple,s["data"])
    vals=[smap.get(c,-20.0) for c in TARGETS]
    top=float(vals[0])
    sorted_vals=sorted(vals,reverse=True)
    target_margin=sorted_vals[0]-sorted_vals[1] if len(sorted_vals)>1 else 0.0

    u_valid,u_repl=decode_metrics(s["data"],"utf-8")
    b_valid,b_repl=decode_metrics(s["data"],"big5")
    k_valid,k_repl=decode_metrics(s["data"],"euc-kr")
    a=np.frombuffer(s["data"][:4096],dtype=np.uint8)
    high=float(np.mean(a>=128)) if len(a) else 0.0

    # Rank positions retain information even when a hard class is outside top-5.
    positions=[]
    for target in TARGETS:
        try: positions.append(float(rank.index(target)+1))
        except ValueError: positions.append(99.0)

    return np.asarray(
        vals
        + positions
        + [target_margin,u_valid,u_repl,b_valid,b_repl,k_valid,k_repl,high,float(len(a))],
        dtype=np.float32
    )


def fit_specialist(train_pool, legacy_X, legacy_y, legacy_b):
    Xrows=[]; yrows=[]
    for inner in range(INNER_FOLDS):
        itr=[s for s in train_pool if inner_assign(s["warc_path"]) != inner]
        iva=[s for s in train_pool if inner_assign(s["warc_path"]) == inner and s["route"]=="RL"]
        itr=lc.deterministic_nested_subset(itr,TRAIN_N)
        models=lc.fit_route_models(itr,legacy_X,legacy_y,legacy_b)
        model=models.get("RL")
        if model is None: continue
        model_classes=set(model[1].classes_)
        for s in iva:
            if s["label"] not in model_classes:
                continue
            Xrows.append(specialist_feature(model,s))
            yrows.append(s["label"] if s["label"] in TARGETS else OTHER)

    if len(set(yrows)) < 2:
        return None

    X=np.stack(Xrows); y=np.asarray(yrows,dtype=object)
    sc=StandardScaler()
    Xs=sc.fit_transform(X)
    clf=LogisticRegression(max_iter=2500,C=0.5,solver="lbfgs",class_weight="balanced")
    clf.fit(Xs,y)
    return sc,clf,len(yrows),Counter(yrows)


def apply_specialist(spec, model, s, baseline, threshold):
    if spec is None or s["route"]!="RL" or not baseline:
        return baseline, False
    sc,clf,_,_=spec
    x=specialist_feature(model,s)
    probs=clf.predict_proba(sc.transform(x[None,:]))[0]
    idx=int(np.argmax(probs))
    pred=str(clf.classes_[idx])
    conf=float(probs[idx])
    if pred==OTHER or conf < threshold:
        return baseline, False

    rank,_=calmod.raw_rank_scores(model,s["data"])
    rank=list(rank)
    if pred not in rank:
        return baseline, False

    out=list(baseline)
    if pred in out:
        out.remove(pred)
    out.insert(0,pred)
    return out, out[0] != baseline[0]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    external=reconstruct_external(state)
    legacy_rows=raw_legacy_rows()
    legacy_X=np.stack([base.scorer_features(r["data"]) for r in legacy_rows])
    legacy_y=np.asarray([r["label"] for r in legacy_rows],dtype=object)
    legacy_b=np.asarray([r["route"] for r in legacy_rows],dtype=object)
    classes,families=calmod.build_vocab(legacy_y)

    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}

    totals={t:Counter() for t in THRESHOLDS}
    baseline_total=Counter()
    fold_rows=[]

    for fold in range(OUTER_FOLDS):
        train_pool=[s for s in external if fold_assign[s["warc_path"]] != fold]
        test_rows=[s for s in external if fold_assign[s["warc_path"]] == fold]
        ext_train=lc.deterministic_nested_subset(train_pool,TRAIN_N)

        models=lc.fit_route_models(ext_train,legacy_X,legacy_y,legacy_b)
        cal=calmod.crossfit_calibration_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        triad_cal=tri.crossfit_triad_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        sig_cal=canon.fit_one(canon.SIG_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        gb_cal=canon.fit_one(canon.GB_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        spec=fit_specialist(train_pool,legacy_X,legacy_y,legacy_b)

        row={"fold":fold,"specialist_training_rows":spec[2] if spec else 0,"specialist_training_counts":dict(spec[3]) if spec else {},"thresholds":{}}
        bc=Counter(); tcs={t:Counter() for t in THRESHOLDS}

        for s in test_rows:
            model=models.get(s["route"])
            if model is None or s["label"] not in model[1].classes_:
                continue
            _,hybrid=tri.make_hybrid(model,cal,s,classes,families)
            gated=gate050.apply_gated(triad_cal,model,s,hybrid)
            sig=canon.apply_one(sig_cal,canon.SIG_PAIR,model,s,gated)
            baseline=canon.apply_guarded_gb(gb_cal,model,s,sig)
            truth=s["label"]

            hit=int(baseline and baseline[0]==truth)
            bc["n"]+=1; bc["hits"]+=hit
            if s["route"]=="RL":
                bc["rl_n"]+=1; bc["rl_hits"]+=hit

            for threshold in THRESHOLDS:
                out,changed=apply_specialist(spec,model,s,baseline,threshold)
                h=int(out and out[0]==truth)
                tc=tcs[threshold]
                tc["n"]+=1; tc["hits"]+=h
                if s["route"]=="RL":
                    tc["rl_n"]+=1; tc["rl_hits"]+=h
                if changed:
                    tc["changes"]+=1
                    if h>hit: tc["beneficial"]+=1
                    elif h<hit: tc["harmful"]+=1

        baseline_total.update(bc)
        row["baseline"]={"n":bc["n"],"hits":bc["hits"],"rl_n":bc["rl_n"],"rl_hits":bc["rl_hits"]}
        for threshold,tc in tcs.items():
            totals[threshold].update(tc)
            row["thresholds"][str(threshold)]=dict(tc)
        fold_rows.append(row)

    results={}
    for threshold,tc in totals.items():
        results[str(threshold)]={
            "n":tc["n"],"hits":tc["hits"],
            "top1":tc["hits"]/max(1,tc["n"]),
            "rl_n":tc["rl_n"],"rl_hits":tc["rl_hits"],
            "rl_top1":tc["rl_hits"]/max(1,tc["rl_n"]),
            "changes":tc["changes"],"beneficial":tc["beneficial"],"harmful":tc["harmful"],
            "delta_hits":tc["hits"]-baseline_total["hits"],
            "delta_rl_hits":tc["rl_hits"]-baseline_total["rl_hits"],
        }

    print(json.dumps({
        "phase":"r12_rl_candidate_specialist",
        "targets":TARGETS,
        "baseline":{
            "n":baseline_total["n"],"hits":baseline_total["hits"],
            "top1":baseline_total["hits"]/max(1,baseline_total["n"]),
            "rl_n":baseline_total["rl_n"],"rl_hits":baseline_total["rl_hits"],
            "rl_top1":baseline_total["rl_hits"]/max(1,baseline_total["rl_n"]),
        },
        "results":results,
        "folds":fold_rows,
        "selection_rule":"Advance only if overall and RL hits both improve, beneficial changes exceed harmful changes, and gains are not confined to one fold.",
        "methodology":{
            "specialist_crossfit_training":True,
            "locked_base_features":True,
            "locked_hmt768":True,
            "existing_downstream_locked":True,
            "specialist_rl_only":True,
            "cc_main_2026_30_used":False,
        }
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
