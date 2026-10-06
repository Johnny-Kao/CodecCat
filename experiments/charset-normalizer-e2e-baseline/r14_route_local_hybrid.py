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

OUTER_FOLDS = 4
TRAIN_N = 300
R12_THRESHOLD = 0.65
MODES = ("A_baseline", "HB_route_generic", "HC_route_encoding")


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


B_EXTRA = len(r13.generic_order_features(b"CodecCat probe"))
C_EXTRA = len(r13.encoding_aware_features(b"CodecCat probe"))


def make_scorer(mode):
    def scorer(data: bytes):
        b=r13.baseline_features(data)
        if mode=="A_baseline":
            return b
        route=base.route_bucket(data)
        if mode=="HB_route_generic":
            extra=r13.generic_order_features(data) if route in ("U","RH") else np.zeros(B_EXTRA,dtype=np.float32)
            return np.concatenate([b,extra])
        if mode=="HC_route_encoding":
            extra=r13.encoding_aware_features(data) if route in ("U","RH") else np.zeros(C_EXTRA,dtype=np.float32)
            return np.concatenate([b,extra])
        raise ValueError(mode)
    return scorer


def feature_cost_us(scorer, samples, repeats=3):
    vals=[]
    for _ in range(repeats):
        t=time.perf_counter_ns()
        for data in samples:
            scorer(data)
        vals.append((time.perf_counter_ns()-t)/1000.0/max(1,len(samples)))
    return float(np.median(vals))


def evaluate_mode(mode, external, fold_assign, legacy_rows):
    scorer=make_scorer(mode)
    old=base.scorer_features
    base.scorer_features=scorer
    try:
        legacy_X=np.stack([scorer(r["data"]) for r in legacy_rows])
        legacy_y=np.asarray([r["label"] for r in legacy_rows],dtype=object)
        legacy_b=np.asarray([r["route"] for r in legacy_rows],dtype=object)
        classes,families=calmod.build_vocab(legacy_y)

        plain=Counter()
        plus=Counter()
        routes_plain={r:Counter() for r in ("U","N","RL","RH")}
        routes_plus={r:Counter() for r in ("U","N","RL","RH")}
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
            spec=r12.fit_specialist(train_pool,legacy_X,legacy_y,legacy_b) if mode!="A_baseline" else None

            fp=Counter(); fs=Counter()
            for s in test_rows:
                model=models.get(s["route"])
                if model is None or s["label"] not in model[1].classes_:
                    continue
                _,hybrid=tri.make_hybrid(model,cal,s,classes,families)
                gated=gate050.apply_gated(triad_cal,model,s,hybrid)
                sig=canon.apply_one(sig_cal,canon.SIG_PAIR,model,s,gated)
                out=canon.apply_guarded_gb(gb_cal,model,s,sig)

                truth=s["label"]
                h=int(out and out[0]==truth)
                plain["n"]+=1; plain["hits"]+=h
                routes_plain[s["route"]]["n"]+=1; routes_plain[s["route"]]["hits"]+=h
                fp["n"]+=1; fp["hits"]+=h

                out2=out
                changed=False
                if spec is not None:
                    out2,changed=r12.apply_specialist(spec,model,s,out,R12_THRESHOLD)
                h2=int(out2 and out2[0]==truth)
                plus["n"]+=1; plus["hits"]+=h2
                routes_plus[s["route"]]["n"]+=1; routes_plus[s["route"]]["hits"]+=h2
                fs["n"]+=1; fs["hits"]+=h2
                if changed:
                    fs["changes"]+=1
                    if h2>h: fs["beneficial"]+=1
                    elif h2<h: fs["harmful"]+=1

            fold_rows.append({
                "fold":fold,
                "plain_hits":fp["hits"],"specialist_hits":fs["hits"],"n":fp["n"],
                "specialist_changes":fs["changes"],
                "specialist_beneficial":fs["beneficial"],
                "specialist_harmful":fs["harmful"],
            })

        samples=[s["data"] for s in external[:128]]
        return {
            "mode":mode,
            "feature_width":len(scorer(b"CodecCat representation probe")),
            "feature_extract_us_per_sample":feature_cost_us(scorer,samples),
            "plain":{
                "n":plain["n"],"hits":plain["hits"],"top1":plain["hits"]/max(1,plain["n"]),
                "routes":{r:{"n":v["n"],"hits":v["hits"],"top1":v["hits"]/max(1,v["n"])} for r,v in routes_plain.items() if v["n"]},
            },
            "plus_r12":{
                "enabled":mode!="A_baseline",
                "threshold":R12_THRESHOLD,
                "n":plus["n"],"hits":plus["hits"],"top1":plus["hits"]/max(1,plus["n"]),
                "routes":{r:{"n":v["n"],"hits":v["hits"],"top1":v["hits"]/max(1,v["n"])} for r,v in routes_plus.items() if v["n"]},
            },
            "folds":fold_rows,
        }
    finally:
        base.scorer_features=old


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    external=reconstruct_external(state)
    legacy_rows=raw_legacy_rows()
    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}

    results={m:evaluate_mode(m,external,fold_assign,legacy_rows) for m in MODES}
    base_hits=results["A_baseline"]["plain"]["hits"]
    base_acc=results["A_baseline"]["plain"]["top1"]
    base_cost=results["A_baseline"]["feature_extract_us_per_sample"]

    variants={}
    for mode,res in results.items():
        variants[mode+"_plain"]={
            "hits":res["plain"]["hits"],
            "top1":res["plain"]["top1"],
            "delta_hits":res["plain"]["hits"]-base_hits,
            "delta_pp":(res["plain"]["top1"]-base_acc)*100.0,
            "feature_cost_ratio":res["feature_extract_us_per_sample"]/max(1e-9,base_cost),
            "routes":res["plain"]["routes"],
        }
        if res["plus_r12"]["enabled"]:
            variants[mode+"_plus_r12"]={
                "hits":res["plus_r12"]["hits"],
                "top1":res["plus_r12"]["top1"],
                "delta_hits":res["plus_r12"]["hits"]-base_hits,
                "delta_pp":(res["plus_r12"]["top1"]-base_acc)*100.0,
                "feature_cost_ratio_excluding_r12":res["feature_extract_us_per_sample"]/max(1e-9,base_cost),
                "routes":res["plus_r12"]["routes"],
            }

    print(json.dumps({
        "phase":"r14_route_local_hybrid_pareto",
        "variants":variants,
        "raw_results":results,
        "selection_rule":"Prefer the cheapest variant that produces a stable paired accuracy gain across folds; R12 threshold is frozen at 0.65 and is not tuned here.",
        "methodology":{
            "same_rows":True,
            "same_fold_assignment":True,
            "full_downstream_refit_per_arm":True,
            "U_RH_only_receive_extra_representation":True,
            "RL_N_extra_dimensions_zeroed":True,
            "r12_threshold_frozen":0.65,
            "cc_main_2026_30_used":False
        }
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
