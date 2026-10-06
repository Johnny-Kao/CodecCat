from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict

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
import r16a_fused_full_b as r16

OUTER_FOLDS=4
TRAIN_N=300
R12_THRESHOLD=0.65
FRACTIONS=(0.10,0.25,0.50,0.75,1.00)


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


def with_scorer(scorer,fn):
    old=base.scorer_features
    base.scorer_features=scorer
    try:
        return fn()
    finally:
        base.scorer_features=old


def legacy_arrays(scorer,legacy_rows):
    X=np.stack([scorer(r["data"]) for r in legacy_rows])
    y=np.asarray([r["label"] for r in legacy_rows],dtype=object)
    b=np.asarray([r["route"] for r in legacy_rows],dtype=object)
    return X,y,b


def fit_stack(scorer,train_pool,legacy_X,legacy_y,legacy_b,with_rl_specialist):
    def _fit():
        classes,families=calmod.build_vocab(legacy_y)
        ext_train=lc.deterministic_nested_subset(train_pool,TRAIN_N)
        models=lc.fit_route_models(ext_train,legacy_X,legacy_y,legacy_b)
        cal=calmod.crossfit_calibration_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        triad_cal=tri.crossfit_triad_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        sig_cal=canon.fit_one(canon.SIG_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        gb_cal=canon.fit_one(canon.GB_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        spec=r12.fit_specialist(train_pool,legacy_X,legacy_y,legacy_b) if with_rl_specialist else None
        return {
            "scorer":scorer,"models":models,"cal":cal,"triad":triad_cal,
            "sig":sig_cal,"gb":gb_cal,"spec":spec,
            "classes":classes,"families":families,
        }
    return with_scorer(scorer,_fit)


def predict(stack,s,apply_rl_specialist):
    def _predict():
        model=stack["models"].get(s["route"])
        if model is None:
            return None,None
        rank,scores=calmod.raw_rank_scores(model,s["data"])
        margin=float(scores[0]-scores[1]) if len(scores)>=2 else float("inf")
        _,hybrid=tri.make_hybrid(model,stack["cal"],s,stack["classes"],stack["families"])
        gated=gate050.apply_gated(stack["triad"],model,s,hybrid)
        sig=canon.apply_one(stack["sig"],canon.SIG_PAIR,model,s,gated)
        out=canon.apply_guarded_gb(stack["gb"],model,s,sig)
        if apply_rl_specialist:
            out,_=r12.apply_specialist(stack["spec"],model,s,out,R12_THRESHOLD)
        return out,margin
    return with_scorer(stack["scorer"],_predict)


def train_margin_thresholds(stack,train_pool):
    margins=defaultdict(list)
    def _collect():
        for s in train_pool:
            if s["route"] not in ("U","RH"):
                continue
            model=stack["models"].get(s["route"])
            if model is None:
                continue
            _,scores=calmod.raw_rank_scores(model,s["data"])
            if len(scores)>=2:
                margins[s["route"]].append(float(scores[0]-scores[1]))
    with_scorer(stack["scorer"],_collect)

    out={}
    for frac in FRACTIONS:
        out[frac]={}
        for route_name in ("U","RH"):
            vals=np.asarray(margins[route_name],dtype=np.float64)
            if frac>=1.0:
                threshold=float("inf")
            elif len(vals):
                threshold=float(np.quantile(vals,frac))
            else:
                threshold=float("-inf")
            out[frac][route_name]=threshold
    return out,{r:len(v) for r,v in margins.items()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    external=reconstruct_external(state); legacy_rows=raw_legacy_rows()

    base_X,legacy_y,legacy_b=legacy_arrays(r13.baseline_features,legacy_rows)
    full_X,full_y,full_b=legacy_arrays(r16.fused_full_b,legacy_rows)
    assert np.array_equal(legacy_y,full_y)
    assert np.array_equal(legacy_b,full_b)

    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}

    baseline=Counter()
    full_all=Counter()
    rh_all=Counter()
    arms={f:Counter() for f in FRACTIONS}
    fold_rows=[]

    for fold in range(OUTER_FOLDS):
        train_pool=[s for s in external if fold_assign[s["warc_path"]]!=fold]
        test_rows=[s for s in external if fold_assign[s["warc_path"]]==fold]

        bstack=fit_stack(r13.baseline_features,train_pool,base_X,legacy_y,legacy_b,True)
        fstack=fit_stack(r16.fused_full_b,train_pool,full_X,full_y,full_b,False)
        thresholds,margin_train_n=train_margin_thresholds(bstack,train_pool)

        fr={
            "fold":fold,
            "margin_training_n":margin_train_n,
            "thresholds":{str(f):thresholds[f] for f in FRACTIONS},
            "baseline":Counter(),"full_all":Counter(),"rh_all":Counter(),
            "arms":{f:Counter() for f in FRACTIONS},
        }

        for s in test_rows:
            bout,bmargin=predict(bstack,s,True)
            if bout is None:
                continue
            truth=s["label"]
            bhit=int(bout and bout[0]==truth)
            baseline["n"]+=1; baseline["hits"]+=bhit
            fr["baseline"]["n"]+=1; fr["baseline"]["hits"]+=bhit

            need_full=s["route"] in ("U","RH")
            fout=None
            if need_full:
                fout,_=predict(fstack,s,False)
            chosen_full=fout if fout is not None else bout
            fhit=int(chosen_full and chosen_full[0]==truth)

            full_all["n"]+=1; full_all["hits"]+=fhit
            fr["full_all"]["n"]+=1; fr["full_all"]["hits"]+=fhit

            rout=(chosen_full if s["route"]=="RH" else bout)
            rhit=int(rout and rout[0]==truth)
            rh_all["n"]+=1; rh_all["hits"]+=rhit
            rh_all["escalated"]+=int(s["route"]=="RH")
            fr["rh_all"]["n"]+=1; fr["rh_all"]["hits"]+=rhit; fr["rh_all"]["escalated"]+=int(s["route"]=="RH")

            for frac in FRACTIONS:
                escalate=(
                    s["route"] in ("U","RH")
                    and bmargin is not None
                    and bmargin <= thresholds[frac][s["route"]]
                )
                out=chosen_full if escalate else bout
                hit=int(out and out[0]==truth)
                c=arms[frac]; fc=fr["arms"][frac]
                c["n"]+=1; c["hits"]+=hit
                fc["n"]+=1; fc["hits"]+=hit
                if s["route"] in ("U","RH"):
                    c["eligible"]+=1; fc["eligible"]+=1
                if escalate:
                    c["escalated"]+=1; fc["escalated"]+=1
                    if hit>bhit:
                        c["beneficial"]+=1; fc["beneficial"]+=1
                    elif hit<bhit:
                        c["harmful"]+=1; fc["harmful"]+=1
                    elif fout is not None and bout and fout and fout[0]!=bout[0]:
                        c["neutral_change"]+=1; fc["neutral_change"]+=1

        def pack(c):
            return {
                "n":c["n"],"hits":c["hits"],"top1":c["hits"]/max(1,c["n"]),
                "escalated":c["escalated"],"eligible":c["eligible"],
                "escalation_rate_all":c["escalated"]/max(1,c["n"]),
                "escalation_rate_eligible":c["escalated"]/max(1,c["eligible"]),
                "beneficial":c["beneficial"],"harmful":c["harmful"],"neutral_change":c["neutral_change"],
            }
        fold_rows.append({
            "fold":fold,
            "margin_training_n":margin_train_n,
            "thresholds":{str(f):thresholds[f] for f in FRACTIONS},
            "baseline":pack(fr["baseline"]),
            "full_all":pack(fr["full_all"]),
            "rh_all":pack(fr["rh_all"]),
            "arms":{str(f):pack(fr["arms"][f]) for f in FRACTIONS},
        })

    def pack(c):
        return {
            "n":c["n"],"hits":c["hits"],"top1":c["hits"]/max(1,c["n"]),
            "delta_hits_vs_baseline":c["hits"]-baseline["hits"],
            "escalated":c["escalated"],"eligible":c["eligible"],
            "escalation_rate_all":c["escalated"]/max(1,c["n"]),
            "escalation_rate_eligible":c["escalated"]/max(1,c["eligible"]),
            "beneficial":c["beneficial"],"harmful":c["harmful"],"neutral_change":c["neutral_change"],
        }

    print(json.dumps({
        "phase":"r17b_ambiguity_gated_full_b",
        "baseline":{"n":baseline["n"],"hits":baseline["hits"],"top1":baseline["hits"]/max(1,baseline["n"])},
        "full_all":{"n":full_all["n"],"hits":full_all["hits"],"top1":full_all["hits"]/max(1,full_all["n"]),"delta_hits_vs_baseline":full_all["hits"]-baseline["hits"]},
        "rh_all":pack(rh_all),
        "arms":{str(f):pack(arms[f]) for f in FRACTIONS},
        "folds":fold_rows,
        "gate_definition":"For each outer fold and each U/RH route separately, escalate when baseline raw top1-top2 margin is below the training-side quantile for the pre-registered fraction. Held-out truth is never used to set thresholds.",
        "selection_note":"This is a Pareto diagnostic on the development corpus, not a release threshold selection. Any chosen gate must be frozen before fresh-crawl validation.",
        "methodology":{"same_outer_folds":True,"thresholds_use_training_margin_distribution_only":True,"r12_rl_threshold":0.65,"cc_main_2026_30_used":False}
    },indent=2,sort_keys=True))


if __name__=="__main__": main()
