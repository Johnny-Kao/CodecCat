"""One-shot frozen-weight RH representation A/B (no threshold fitting)."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import chardet
from charset_normalizer import from_bytes
from codeccat import load_bundle
from codeccat.runtime import Runtime, _signals, family
import codeccat.runtime as runtime_mod
from h2_fused_candidate_mask_ab import collect, normalize

SHA="e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54"
EXPLORATORY=("CC-MAIN-2025-13","CC-MAIN-2025-38","CC-MAIN-2025-51","CC-MAIN-2026-04")
SEED=20261014
ORIG=runtime_mod.hmt768_array

def spaced768(data:bytes):
    # Keep original HMT representation for short inputs.
    if len(data)<=768 or _signals(data).route != "RH":
        return ORIG(data)
    arr=np.frombuffer(data,dtype=np.uint8)
    # Fixed, evenly-spaced 768 byte positions across the entire message.
    idx=(np.arange(768,dtype=np.int64)*len(arr))//768
    return arr[idx]

def policies(rt, data):
    control=rt.rank(data)[0]
    with patch.object(runtime_mod,"hmt768_array",spaced768):
        alternate=rt.rank(data)[0]
    return control,alternate

def run(args):
    from codeccat.runtime import _signals
    import requests
    path=Path(args.model)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA
    rt=Runtime(load_bundle(path))
    for raw in (b"hello",b"\xef\xbb\xbfhello",b"\x80\x81"*512,b"\x00\xff"*512):
        c,a=policies(rt,raw)
        assert c in rt.bundle.classes and a in rt.bundle.classes
        assert c==rt.rank(raw)[0]
    print("REPRESENTATION_PREFLIGHT_PASS",flush=True)
    catalog=requests.get("https://index.commoncrawl.org/collinfo.json",timeout=40).json()
    pool=sorted({x["id"] for x in catalog if str(x.get("id","")).startswith("CC-MAIN-2024-")})
    if len(pool)<3: raise RuntimeError("Insufficient independent 2024 indexes")
    prospective=tuple(sorted(random.Random(SEED).sample(pool,3)))
    print("FROZEN_2024_CRAWLS",prospective,flush=True)
    seen=set()
    rows=[]
    for crawl in EXPLORATORY+prospective:
        samples,_,stats=collect(crawl)
        for sample in samples:
            digest=hashlib.sha256(sample["data"]).hexdigest()
            if digest in seen:continue
            seen.add(digest)
            rows.append({"crawl":crawl,"data":sample["data"],"truth":normalize(sample["label"])})
        print("COLLECTED",crawl,len(samples),flush=True)
    assert len(rows)>=500
    reports=[]
    for item in rows:
        data=item["data"]
        c,a=policies(rt,data)
        ch=chardet.detect(data).get("encoding")
        cn=from_bytes(data).best()
        reports.append({"crawl":item["crawl"],"group":"exploratory" if item["crawl"] in EXPLORATORY else "prospective",
                        "sha":hashlib.sha256(data).hexdigest(),"route":_signals(data).route,
                        "family":family(item["truth"]),"truth":item["truth"],
                        "control":c,"uniform":a,"chardet7":normalize(ch),
                        "charset_normalizer":normalize(cn.encoding) if cn else None})
    def summarize(items):
        n=len(items)
        x={"n":n}
        for key in ("control","uniform","chardet7","charset_normalizer"):
            hit=sum(v[key]==v["truth"] for v in items)
            x[key]={"hits":hit,"accuracy":hit/n if n else None}
        x["changed"]=sum(v["control"]!=v["uniform"] for v in items)
        x["beneficial"]=sum(v["control"]!=v["truth"] and v["uniform"]==v["truth"] for v in items)
        x["harmful"]=sum(v["control"]==v["truth"] and v["uniform"]!=v["truth"] for v in items)
        return x
    result={"candidate":"RH deterministic evenly-spaced 768 positions; frozen R21 models and decisions",
            "random_seed":SEED,"prospective_crawls":prospective,"no_retraining":True,
            "limitation":"2024 sample is prospective for this A/B but original training-set overlap unverified",
            "overall":summarize(reports),
            "groups":{k:summarize([r for r in reports if r["group"]==k]) for k in ("exploratory","prospective")},
            "by_crawl":{k:summarize([r for r in reports if r["crawl"]==k]) for k in sorted({r["crawl"] for r in reports})},
            "prospective_rh":summarize([r for r in reports if r["group"]=="prospective" and r["route"]=="RH"]),
            "prospective_cjk":summarize([r for r in reports if r["group"]=="prospective" and r["family"]=="cjk"])}
    # Warm inference-only timing on the exact same bytes, no imports/network in loops.
    data=[r["data"] for r in rows]
    for policy in ("control","uniform"):
        times=[]
        for repeat in range(5):
            ordering=data if repeat%2==0 else list(reversed(data))
            start=time.perf_counter_ns()
            if policy=="control":
                for d in ordering:rt.rank(d)
            else:
                with patch.object(runtime_mod,"hmt768_array",spaced768):
                    for d in ordering:rt.rank(d)
            times.append((time.perf_counter_ns()-start)/len(ordering))
        result.setdefault("timing_ns_per_sample",{})[policy]=statistics.median(times)
    result["speed_delta_fraction"]=result["timing_ns_per_sample"]["uniform"]/result["timing_ns_per_sample"]["control"]-1
    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=True)
    (output/"results.json").write_text(json.dumps(result,indent=2))
    (output/"sample-predictions.json").write_text(json.dumps(reports))
    print("ARCH_RESULT_START",flush=True)
    print(json.dumps(result,indent=2),flush=True)
    print("ARCH_RESULT_END",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--out",default="research-results/architecture")
    run(p.parse_args())
