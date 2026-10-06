from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import unicodedata
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

OUTER_FOLDS = 4
TRAIN_N = 300
MODES = ("A_baseline", "B_generic_order", "C_encoding_aware")

MULTIBYTE = ("utf-8", "big5", "gb18030", "shift_jis", "euc_jp", "euc_kr")
SCRIPT_CODECS = ("cp1251", "cp1250", "cp1252", "iso8859_2", "iso8859_5")

def hmt768(data: bytes) -> bytes:
    if len(data) <= 768:
        return data
    mid = len(data)//2
    return data[:256] + data[mid-128:mid+128] + data[-256:]


def baseline_features(data: bytes) -> np.ndarray:
    a=np.frombuffer(hmt768(data),dtype=np.uint8)
    n=max(1,len(a))
    unigram=np.bincount(a,minlength=256).astype(np.float32)/n
    wrapped=np.zeros(256,dtype=np.float32)
    if len(a)>=2:
        bins=a[:-1]+a[1:]
        wrapped=np.bincount(bins,minlength=256).astype(np.float32)/(len(a)-1)
    scalars=np.asarray([
        math.log2(n+1)/16.0,
        float(np.mean(a>=128)) if len(a) else 0.0,
        float(np.mean(a==0)) if len(a) else 0.0,
        float(np.mean((a>=32)&(a<=126))) if len(a) else 0.0,
        float(np.mean(a==10)) if len(a) else 0.0,
        float(np.mean(a==13)) if len(a) else 0.0,
    ],dtype=np.float32)
    return np.concatenate([unigram,wrapped,scalars])


def generic_order_features(data: bytes) -> np.ndarray:
    a=np.frombuffer(hmt768(data),dtype=np.uint8)
    if len(a)<2:
        diff=np.zeros(256,dtype=np.float32)
        xor=np.zeros(64,dtype=np.float32)
    else:
        x=a[:-1].astype(np.int16); y=a[1:].astype(np.int16)
        db=((y-x)&255).astype(np.intp)
        diff=np.bincount(db,minlength=256).astype(np.float32)/(len(a)-1)
        xb=np.bitwise_xor(a[:-1],a[1:])>>2
        xor=np.bincount(xb,minlength=64).astype(np.float32)/(len(a)-1)

    # Position-aware coarse high-byte distribution: 3 chunks x 16 bins.
    pos=[]
    sample=hmt768(data)
    cuts=(sample[:256], sample[256:512], sample[512:768])
    for chunk in cuts:
        c=np.frombuffer(chunk,dtype=np.uint8)
        high=c[c>=128]
        if len(high):
            idx=((high.astype(np.uint16)-128)>>3).astype(np.intp)
            pos.append(np.bincount(idx,minlength=16).astype(np.float32)/len(high))
        else:
            pos.append(np.zeros(16,dtype=np.float32))
    return np.concatenate([diff,xor,*pos])


def decode_metrics(data: bytes, enc: str):
    sample=data[:4096]
    try:
        text=sample.decode(enc,"strict")
        valid=1.0
        repl=0.0
    except Exception:
        valid=0.0
        try:
            text=sample.decode(enc,"replace")
            repl=text.count("\ufffd")/max(1,len(text))
        except Exception:
            text=""
            repl=1.0
    return valid,repl,text


def script_stats(text: str):
    if not text:
        return (0.0,0.0,0.0,0.0)
    n=max(1,len(text))
    letters=0; cyr=0; latin=0; ctrl=0
    for ch in text:
        o=ord(ch)
        cat=unicodedata.category(ch)
        if cat.startswith("L"):
            letters+=1
            name=unicodedata.name(ch,"")
            cyr += int("CYRILLIC" in name)
            latin += int("LATIN" in name)
        ctrl += int(cat.startswith("C") and ch not in "\r\n\t")
    return (letters/n,cyr/n,latin/n,ctrl/n)


def encoding_aware_features(data: bytes) -> np.ndarray:
    out=[]
    for enc in MULTIBYTE:
        valid,repl,text=decode_metrics(data,enc)
        letters,cyr,latin,ctrl=script_stats(text)
        out.extend([valid,repl,letters,cyr,latin,ctrl])

    # Single-byte codecs are structurally permissive, so use only cheap decoded-script
    # composition rather than meaningless strict-valid flags.
    for enc in SCRIPT_CODECS:
        _,_,text=decode_metrics(data,enc)
        letters,cyr,latin,ctrl=script_stats(text)
        out.extend([letters,cyr,latin,ctrl])
    return np.asarray(out,dtype=np.float32)


def make_features(mode: str):
    def scorer(data: bytes) -> np.ndarray:
        b=baseline_features(data)
        if mode=="A_baseline":
            return b
        if mode=="B_generic_order":
            return np.concatenate([b,generic_order_features(data)])
        if mode=="C_encoding_aware":
            return np.concatenate([b,encoding_aware_features(data)])
        raise ValueError(mode)
    return scorer


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


def feature_cost_us(scorer, samples, repeats=3):
    # Same samples for all arms; report median per-sample feature extraction only.
    vals=[]
    for _ in range(repeats):
        t=time.perf_counter_ns()
        for data in samples:
            scorer(data)
        vals.append((time.perf_counter_ns()-t)/1000.0/max(1,len(samples)))
    return float(np.median(vals))


def evaluate_mode(mode, external, fold_assign, legacy_rows):
    scorer=make_features(mode)
    old=base.scorer_features
    base.scorer_features=scorer
    try:
        legacy_X=np.stack([scorer(r["data"]) for r in legacy_rows])
        legacy_y=np.asarray([r["label"] for r in legacy_rows],dtype=object)
        legacy_b=np.asarray([r["route"] for r in legacy_rows],dtype=object)
        classes,families=calmod.build_vocab(legacy_y)

        pooled=Counter()
        routes={r:Counter() for r in ("U","N","RL","RH")}
        folds=[]
        for fold in range(OUTER_FOLDS):
            train_pool=[s for s in external if fold_assign[s["warc_path"]] != fold]
            test_rows=[s for s in external if fold_assign[s["warc_path"]] == fold]
            ext_train=lc.deterministic_nested_subset(train_pool,TRAIN_N)

            tf=time.perf_counter()
            models=lc.fit_route_models(ext_train,legacy_X,legacy_y,legacy_b)
            cal=calmod.crossfit_calibration_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
            triad_cal=tri.crossfit_triad_rows(train_pool,legacy_X,legacy_y,legacy_b,classes,families)
            sig_cal=canon.fit_one(canon.SIG_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)
            gb_cal=canon.fit_one(canon.GB_PAIR,train_pool,legacy_X,legacy_y,legacy_b,classes,families)

            c=Counter(); rc={r:Counter() for r in ("U","N","RL","RH")}
            for s in test_rows:
                model=models.get(s["route"])
                if model is None or s["label"] not in model[1].classes_:
                    continue
                _,hybrid=tri.make_hybrid(model,cal,s,classes,families)
                gated=gate050.apply_gated(triad_cal,model,s,hybrid)
                sig=canon.apply_one(sig_cal,canon.SIG_PAIR,model,s,gated)
                out=canon.apply_guarded_gb(gb_cal,model,s,sig)
                hit=int(out and out[0]==s["label"])
                c["n"]+=1; c["hits"]+=hit
                rc[s["route"]]["n"]+=1; rc[s["route"]]["hits"]+=hit

            pooled.update(c)
            for r,v in rc.items(): routes[r].update(v)
            folds.append({"fold":fold,"n":c["n"],"hits":c["hits"],"top1":c["hits"]/max(1,c["n"]),"training_seconds":time.perf_counter()-tf})

        sample_bytes=[s["data"] for s in external[:128]]
        return {
            "mode":mode,
            "feature_width":len(scorer(b"CodecCat representation probe")),
            "feature_extract_us_per_sample":feature_cost_us(scorer,sample_bytes),
            "n":pooled["n"],"hits":pooled["hits"],"top1":pooled["hits"]/max(1,pooled["n"]),
            "routes":{r:{"n":v["n"],"hits":v["hits"],"top1":v["hits"]/max(1,v["n"])} for r,v in routes.items() if v["n"]},
            "folds":folds,
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
    a=results["A_baseline"]
    deltas={}
    for m,r in results.items():
        deltas[m]={
            "hits":r["hits"]-a["hits"],
            "top1_pp":(r["top1"]-a["top1"])*100.0,
            "feature_cost_ratio":r["feature_extract_us_per_sample"]/max(1e-9,a["feature_extract_us_per_sample"]),
            "rl_hits":r["routes"].get("RL",{}).get("hits",0)-a["routes"].get("RL",{}).get("hits",0),
            "rh_hits":r["routes"].get("RH",{}).get("hits",0)-a["routes"].get("RH",{}).get("hits",0),
        }

    print(json.dumps({
        "phase":"r13_representation_sufficiency_abc",
        "results":results,
        "paired_deltas_vs_A":deltas,
        "interpretation":{
            "B_positive":"Generic representation is the main information bottleneck.",
            "C_positive_B_flat":"Encoding-specific structural/decoded-script information is missing from the current representation.",
            "B_and_C_flat":"The ceiling is more likely classifier/route/model-family related than representation-only.",
            "C_cost_note":"Feature extraction cost is diagnostic Python cost, not an optimized production implementation."
        },
        "methodology":{
            "same_rows":True,
            "same_fold_assignment":True,
            "same_classifier_family":True,
            "full_downstream_refit_per_arm":True,
            "no_r12_specialist":True,
            "cc_main_2026_30_used":False
        }
    },indent=2,sort_keys=True))

if __name__=="__main__":
    main()
