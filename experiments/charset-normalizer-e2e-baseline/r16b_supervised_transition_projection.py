from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter

import joblib
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

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
DIMS=(3,8,16,32)


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


def pair_ids(data: bytes) -> np.ndarray:
    a=np.frombuffer(r13.hmt768(data),dtype=np.uint8)
    if len(a)<2:
        return np.empty(0,dtype=np.int32)
    return ((a[:-1].astype(np.int32)<<8) | a[1:].astype(np.int32)).astype(np.int32,copy=False)


def pair_sparse_row(data: bytes):
    ids=pair_ids(data)
    if len(ids)==0:
        return sparse.csr_matrix((1,65536),dtype=np.float32)
    uniq,cnt=np.unique(ids,return_counts=True)
    vals=(cnt.astype(np.float32)/len(ids)).astype(np.float32,copy=False)
    indptr=np.asarray([0,len(uniq)],dtype=np.int32)
    return sparse.csr_matrix((vals,uniq.astype(np.int32,copy=False),indptr),shape=(1,65536),dtype=np.float32)


def fit_supervised_projection(train_rows,legacy_rows):
    rows=[r for r in legacy_rows if r["route"] in ("U","RH")]
    rows += [r for r in train_rows if r["route"] in ("U","RH")]
    y=np.asarray([r["label"] for r in rows],dtype=object)
    X=sparse.vstack([pair_sparse_row(r["data"]) for r in rows],format="csr")

    classes=np.unique(y)
    centroids=np.zeros((len(classes),65536),dtype=np.float32)
    counts=np.zeros(len(classes),dtype=np.int32)
    for i,c in enumerate(classes):
        idx=np.where(y==c)[0]
        counts[i]=len(idx)
        centroids[i]=np.asarray(X[idx].mean(axis=0),dtype=np.float32).ravel()

    global_mean=np.average(centroids,axis=0,weights=counts).astype(np.float32)
    between=(centroids-global_mean[None,:]) * np.sqrt(counts.astype(np.float32))[:,None]
    _,singular,vt=randomized_svd(
        between,
        n_components=max(DIMS),
        n_iter=7,
        random_state=0,
    )
    return vt.astype(np.float32,copy=False),singular.astype(np.float32,copy=False),len(classes)


def make_scorer(projection,dim):
    p=projection[:dim]
    def scorer(data):
        b=r13.baseline_features(data)
        if base.route_bucket(data) not in ("U","RH"):
            return np.concatenate([b,np.zeros(dim,dtype=np.float32)])
        ids=pair_ids(data)
        if len(ids)==0:
            z=np.zeros(dim,dtype=np.float32)
        else:
            # Equivalent to normalized pair histogram @ projection.T, but directly
            # accumulates lookup columns. This is also the natural C/C++ runtime form.
            z=np.mean(p[:,ids],axis=1,dtype=np.float32)
        return np.concatenate([b,z.astype(np.float32,copy=False)])
    return scorer


def feature_cost_us(fn,samples,repeats=3):
    vals=[]
    for _ in range(repeats):
        t=time.perf_counter_ns()
        for d in samples: fn(d)
        vals.append((time.perf_counter_ns()-t)/1000/max(1,len(samples)))
    return float(np.median(vals))


def evaluate_arm(dim,projection,external,fold,train_pool,test_rows,legacy_rows):
    scorer=make_scorer(projection,dim)
    old=base.scorer_features; base.scorer_features=scorer
    try:
        legacy_X=np.stack([scorer(r["data"]) for r in legacy_rows])
        legacy_y=np.asarray([r["label"] for r in legacy_rows],dtype=object)
        legacy_b=np.asarray([r["route"] for r in legacy_rows],dtype=object)
        classes,families=calmod.build_vocab(legacy_y)

        ext_train=lc.deterministic_nested_subset(train_pool,TRAIN_N)
        models=lc.fit_route_models(ext_train,legacy_X,legacy_y,legacy_b)
        cal=calmod.crossfit_calibration_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        triad_cal=tri.crossfit_triad_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        sig_cal=canon.fit_one(canon.SIG_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        gb_cal=canon.fit_one(canon.GB_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        spec=r12.fit_specialist(train_pool,legacy_X,legacy_y,legacy_b)

        c=Counter(); routes={r:Counter() for r in ("U","N","RL","RH")}
        for s in test_rows:
            model=models.get(s["route"])
            if model is None or s["label"] not in model[1].classes_: continue
            _,hybrid=tri.make_hybrid(model,cal,s,classes,families)
            gated=gate050.apply_gated(triad_cal,model,s,hybrid)
            sig=canon.apply_one(sig_cal,canon.SIG_PAIR,model,s,gated)
            out=canon.apply_guarded_gb(gb_cal,model,s,sig)
            out,_=r12.apply_specialist(spec,model,s,out,R12_THRESHOLD)
            hit=int(out and out[0]==s["label"])
            c["n"]+=1; c["hits"]+=hit
            routes[s["route"]]["n"]+=1; routes[s["route"]]["hits"]+=hit
        return {
            "fold":fold,"dim":dim,"n":c["n"],"hits":c["hits"],"top1":c["hits"]/max(1,c["n"]),
            "routes":{r:{"n":v["n"],"hits":v["hits"],"top1":v["hits"]/max(1,v["n"])} for r,v in routes.items() if v["n"]},
        },scorer
    finally:
        base.scorer_features=old


def evaluate_baseline(fold,train_pool,test_rows,legacy_rows):
    scorer=r13.baseline_features
    old=base.scorer_features; base.scorer_features=scorer
    try:
        legacy_X=np.stack([scorer(r["data"]) for r in legacy_rows])
        legacy_y=np.asarray([r["label"] for r in legacy_rows],dtype=object)
        legacy_b=np.asarray([r["route"] for r in legacy_rows],dtype=object)
        classes,families=calmod.build_vocab(legacy_y)
        ext_train=lc.deterministic_nested_subset(train_pool,TRAIN_N)
        models=lc.fit_route_models(ext_train,legacy_X,legacy_y,legacy_b)
        cal=calmod.crossfit_calibration_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        triad_cal=tri.crossfit_triad_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        sig_cal=canon.fit_one(canon.SIG_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        gb_cal=canon.fit_one(canon.GB_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)
        spec=r12.fit_specialist(train_pool,legacy_X,legacy_y,legacy_b)
        c=Counter()
        for s in test_rows:
            model=models.get(s["route"])
            if model is None or s["label"] not in model[1].classes_: continue
            _,hybrid=tri.make_hybrid(model,cal,s,classes,families)
            gated=gate050.apply_gated(triad_cal,model,s,hybrid)
            sig=canon.apply_one(sig_cal,canon.SIG_PAIR,model,s,gated)
            out=canon.apply_guarded_gb(gb_cal,model,s,sig)
            out,_=r12.apply_specialist(spec,model,s,out,R12_THRESHOLD)
            c["n"]+=1; c["hits"]+=int(out and out[0]==s["label"])
        return {"fold":fold,"n":c["n"],"hits":c["hits"],"top1":c["hits"]/max(1,c["n"])}
    finally:
        base.scorer_features=old


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    external=reconstruct_external(state); legacy_rows=raw_legacy_rows()
    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}

    pooled={d:Counter() for d in DIMS}; route_pooled={d:{r:Counter() for r in ("U","N","RL","RH")} for d in DIMS}
    folds={d:[] for d in DIMS}; costs={d:[] for d in DIMS}
    baseline=Counter(); baseline_folds=[]
    proj_meta={}
    samples=[s["data"] for s in external[:128]]

    for fold in range(OUTER_FOLDS):
        train_pool=[s for s in external if fold_assign[s["warc_path"]]!=fold]
        test_rows=[s for s in external if fold_assign[s["warc_path"]]==fold]

        br=evaluate_baseline(fold,train_pool,test_rows,legacy_rows)
        baseline_folds.append(br); baseline["n"]+=br["n"]; baseline["hits"]+=br["hits"]

        projection,singular,n_classes=fit_supervised_projection(train_pool,legacy_rows)
        energy=np.cumsum(singular.astype(np.float64)**2)
        energy/=max(1e-30,energy[-1])
        proj_meta[str(fold)]={
            "classes":n_classes,
            "between_class_energy_3":float(energy[2]),
            "between_class_energy_8":float(energy[7]),
            "between_class_energy_16":float(energy[15]),
            "between_class_energy_32":float(energy[31]),
        }

        for dim in DIMS:
            row,scorer=evaluate_arm(dim,projection,external,fold,train_pool,test_rows,legacy_rows)
            folds[dim].append(row)
            pooled[dim]["n"]+=row["n"]; pooled[dim]["hits"]+=row["hits"]
            for r,v in row["routes"].items():
                route_pooled[dim][r]["n"]+=v["n"]; route_pooled[dim][r]["hits"]+=v["hits"]
            costs[dim].append(feature_cost_us(scorer,samples,repeats=1))

    base_acc=baseline["hits"]/max(1,baseline["n"])
    results={}
    for dim in DIMS:
        c=pooled[dim]; acc=c["hits"]/max(1,c["n"])
        results[str(dim)]={
            "dim":dim,"feature_width":518+dim,
            "n":c["n"],"hits":c["hits"],"top1":acc,
            "delta_hits_vs_baseline":c["hits"]-baseline["hits"],
            "delta_pp_vs_baseline":(acc-base_acc)*100.0,
            "median_feature_extract_us_per_sample":float(np.median(costs[dim])),
            "routes":{r:{"n":v["n"],"hits":v["hits"],"top1":v["hits"]/max(1,v["n"])} for r,v in route_pooled[dim].items() if v["n"]},
            "folds":folds[dim],
        }

    print(json.dumps({
        "phase":"r16b_supervised_transition_projection",
        "baseline":{"n":baseline["n"],"hits":baseline["hits"],"top1":base_acc,"folds":baseline_folds},
        "results":results,
        "projection_meta":proj_meta,
        "representation":"Exact 256x256 HMT768 byte-pair histogram -> supervised between-class scatter subspace -> 3/8/16/32D -> existing linear/downstream pipeline.",
        "scientific_note":"Right singular vectors of sqrt(n_c)*(class_centroid-global_centroid) maximize captured between-class scatter. Unlike R15B SVD, labels directly define the projection objective.",
        "runtime_note":"Inference uses direct pair-id lookup and accumulation; projection plus classifier can later be algebraically fused into fixed pair lookup weights in C/C++.",
        "methodology":{"outer_test_not_used_for_projection_fit":True,"projection_fit_legacy_plus_outer_train":True,"same_outer_folds":True,"r12_threshold":0.65,"cc_main_2026_30_used":False}
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
