from __future__ import annotations

import argparse
import hashlib
import json
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
MODES=("baseline","diff256","xor64","pos48","diff_xor","full_B")


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


def split_generic(data):
    g=r13.generic_order_features(data)
    return g[:256],g[256:320],g[320:368]


WIDTH={"baseline":0,"diff256":256,"xor64":64,"pos48":48,"diff_xor":320,"full_B":368}


def make_scorer(mode):
    width=WIDTH[mode]
    def scorer(data):
        b=r13.baseline_features(data)
        if width==0: return b
        if base.route_bucket(data) not in ("U","RH"):
            return np.concatenate([b,np.zeros(width,dtype=np.float32)])
        d,x,p=split_generic(data)
        if mode=="diff256": extra=d
        elif mode=="xor64": extra=x
        elif mode=="pos48": extra=p
        elif mode=="diff_xor": extra=np.concatenate([d,x])
        elif mode=="full_B": extra=np.concatenate([d,x,p])
        else: raise ValueError(mode)
        return np.concatenate([b,extra])
    return scorer


def feature_cost_us(scorer,samples,repeats=3):
    vals=[]
    for _ in range(repeats):
        t=time.perf_counter_ns()
        for d in samples: scorer(d)
        vals.append((time.perf_counter_ns()-t)/1000/max(1,len(samples)))
    return float(np.median(vals))


def evaluate(mode,external,fold_assign,legacy_rows):
    scorer=make_scorer(mode)
    old=base.scorer_features; base.scorer_features=scorer
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
        samples=[s["data"] for s in external[:128]]
        return {
            "mode":mode,"feature_width":len(scorer(b"CodecCat scientific ablation")),
            "feature_extract_us_per_sample":feature_cost_us(scorer,samples),
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
    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}
    results={m:evaluate(m,external,fold_assign,legacy_rows) for m in MODES}
    b=results["baseline"]
    for m,r in results.items():
        r["delta_hits_vs_baseline"]=r["hits"]-b["hits"]
        r["delta_pp_vs_baseline"]=(r["top1"]-b["top1"])*100
        r["cost_ratio_vs_baseline"]=r["feature_extract_us_per_sample"]/max(1e-9,b["feature_extract_us_per_sample"])
    print(json.dumps({
        "phase":"r15a_scientific_feature_ablation",
        "results":results,
        "selection_rule":"Choose the smallest feature group whose paired gain is stable across folds and close to full_B; R12 is frozen at 0.65.",
        "methodology":{"same_rows":True,"same_folds":True,"full_downstream_refit":True,"extra_features_U_RH_only":True,"r12_threshold":0.65,"cc_main_2026_30_used":False}
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
