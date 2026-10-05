from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib

import charset_external_scorer_learning_curve as lc
import charset_external_scorer_domain_shift_ab as base
import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_canonical_guarded_final_validation as canon

SCHEMA_VERSION = 1
OUTER_FOLDS = 4
TRAIN_N = 300

def sample_fingerprint(samples):
    h=hashlib.sha256()
    for s in samples:
        h.update(s["warc_path"].encode())
        h.update(b"\0"); h.update(s["route"].encode())
        h.update(b"\0"); h.update(s["label"].encode())
        h.update(b"\0"); h.update(hashlib.sha256(s["data"]).digest())
    return h.hexdigest()

def build_state():
    t0=time.perf_counter()
    external,paths,stats=base.collect_external()
    legacy_X,legacy_y,legacy_b=base.load_legacy_training()
    classes,families=calmod.build_vocab(legacy_y)
    pathset=sorted(set(s["warc_path"] for s in external))
    fold_assign={
        p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS
        for p in pathset
    }
    folds=[]
    training_ns=0
    for fold in range(OUTER_FOLDS):
        train=[s for s in external if fold_assign[s["warc_path"]]!=fold]
        test=[s for s in external if fold_assign[s["warc_path"]]==fold]
        ext=lc.deterministic_nested_subset(train,TRAIN_N)
        tf=time.perf_counter_ns()
        models=lc.fit_route_models(ext,legacy_X,legacy_y,legacy_b)
        cal=calmod.crossfit_calibration_rows(train,legacy_X,legacy_y,legacy_b,classes,families)
        triad_cal=tri.crossfit_triad_rows(train,legacy_X,legacy_y,legacy_b,classes,families)
        sig_cal=canon.fit_one(canon.SIG_PAIR,train,legacy_X,legacy_y,legacy_b,classes,families)
        gb_cal=canon.fit_one(canon.GB_PAIR,train,legacy_X,legacy_y,legacy_b,classes,families)
        training_ns+=time.perf_counter_ns()-tf
        rows=[s for s in test if models.get(s["route"]) is not None and s["label"] in models[s["route"]][1].classes_]
        folds.append({
            "fold":fold,"models":models,"cal":cal,"triad_cal":triad_cal,
            "sig_cal":sig_cal,"gb_cal":gb_cal,"rows":rows,
        })
    state={
        "schema_version":SCHEMA_VERSION,
        "corpus_fingerprint":sample_fingerprint(external),
        "external_n":len(external),
        "paths":paths,
        "collection_stats":dict(stats),
        "classes":classes,
        "families":families,
        "folds":folds,
        "training_seconds":training_ns/1e9,
        "build_seconds":time.perf_counter()-t0,
    }
    return state

def validate_state(state):
    assert state["schema_version"]==SCHEMA_VERSION
    assert len(state["folds"])==OUTER_FOLDS
    assert sum(len(f["rows"]) for f in state["folds"])==418

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",required=True)
    args=ap.parse_args()
    path=Path(args.state)
    if path.exists():
        state=joblib.load(path)
        validate_state(state)
        status="hit"
    else:
        state=build_state()
        validate_state(state)
        path.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(state,path,compress=3)
        status="miss_built"
    print(json.dumps({
        "phase":"cached_fold_state",
        "cache_status":status,
        "schema_version":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "external_n":state["external_n"],
        "evaluated_n":sum(len(f["rows"]) for f in state["folds"]),
        "training_seconds":state.get("training_seconds",0.0),
        "build_seconds":state.get("build_seconds",0.0),
        "state_bytes":path.stat().st_size,
        "collection_stats":state.get("collection_stats",{}),
    },indent=2,sort_keys=True))

if __name__=="__main__":
    main()
