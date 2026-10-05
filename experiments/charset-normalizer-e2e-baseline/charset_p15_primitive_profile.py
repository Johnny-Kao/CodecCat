from __future__ import annotations

import argparse
import json
import math
import time

import joblib
import numpy as np

import charset_p2_multihop_kernel_tournament as p2
import charset_p12_normalized_count_tournament as p12

REPEATS=100

def bench(rows, fn):
    n=len(rows); sink=0
    t0=time.perf_counter_ns()
    for _ in range(REPEATS):
        for i in range(n):
            x=fn(i)
            if isinstance(x,np.ndarray): sink+=x.size
            elif x is not None: sink+=1
    return (time.perf_counter_ns()-t0)/(REPEATS*n),sink

def scalar_stats(a,counts):
    n=max(1,len(a))
    if len(a):
        inv=1.0/len(a)
        return (
            math.log2(n+1)/16.0,
            float(counts[128:].sum()*inv),
            float(counts[0]*inv),
            float(counts[32:127].sum()*inv),
            float(counts[10]*inv),
            float(counts[13]*inv),
        )
    return (math.log2(n+1)/16.0,0.0,0.0,0.0,0.0,0.0)

def finish(a,counts,bc):
    n=max(1,len(a))
    out=np.empty(518,dtype=np.float32)
    out[:256]=counts; out[:256]*=1.0/n
    if len(a)>=2:
        out[256:512]=bc; out[256:512]*=1.0/(len(a)-1)
    else:
        out[256:512]=0.0
    out[512:]=scalar_stats(a,counts)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    assert state["schema_version"]==1
    all_rows=[]
    for fd in state["folds"]:
        for s in fd["rows"]:
            all_rows.append((fd,s))

    sampled=[p2.hmt768(s["data"]) for _,s in all_rows]
    arrays=[np.frombuffer(x,dtype=np.uint8) for x in sampled]
    counts=[np.bincount(a,minlength=256) for a in arrays]
    bins=[a[:-1]+a[1:] if len(a)>=2 else np.empty(0,dtype=np.uint8) for a in arrays]
    bcounts=[np.bincount(b,minlength=256) if len(b) else np.zeros(256,dtype=np.int64) for b in bins]

    feature={}
    feature["hmt768"],_=bench(all_rows,lambda i:p2.hmt768(all_rows[i][1]["data"]))
    feature["frombuffer"],_=bench(all_rows,lambda i:np.frombuffer(sampled[i],dtype=np.uint8))
    feature["unigram_bincount"],_=bench(all_rows,lambda i:np.bincount(arrays[i],minlength=256))
    feature["bigram_add"],_=bench(all_rows,lambda i:arrays[i][:-1]+arrays[i][1:] if len(arrays[i])>=2 else np.empty(0,dtype=np.uint8))
    feature["bigram_bincount"],_=bench(all_rows,lambda i:np.bincount(bins[i],minlength=256) if len(bins[i]) else np.zeros(256,dtype=np.int64))
    feature["scalar_stats"],_=bench(all_rows,lambda i:scalar_stats(arrays[i],counts[i]))
    feature["finish_counts"],_=bench(all_rows,lambda i:finish(arrays[i],counts[i],bcounts[i]))
    feature["full_feature"],_=bench(all_rows,lambda i:p12.features_kernel(all_rows[i][1]["data"],"assign_multiply","assign_multiply"))

    # Score/order primitives use each fold's route-specific fused model.
    xs=[]; fused_rows=[]; x64=[]; raw=[]
    for fd,s in all_rows:
        fm=p2.fuse_scaler_linear(fd["models"][s["route"]])
        x=p12.features_kernel(s["data"],"assign_multiply","assign_multiply")
        xs.append(x); fused_rows.append(fm)
        xx=np.asarray(x,dtype=np.float64); x64.append(xx)
        classes,w,b=fm
        z=xx@w.T+b
        if len(classes)==2 and z.shape==(1,):
            q=float(z[0]); z=np.asarray([-q,q],dtype=np.float64)
        else:
            z=np.asarray(z,dtype=np.float64)
        raw.append(z)

    score={}
    score["cast_float64"],_=bench(all_rows,lambda i:np.asarray(xs[i],dtype=np.float64))
    def dot(i):
        classes,w,b=fused_rows[i]
        return x64[i]@w.T+b
    score["matmul_add"],_=bench(all_rows,dot)
    score["argsort_reverse"],_=bench(all_rows,lambda i:raw[i].argsort()[::-1])
    score["full_score"],_=bench(all_rows,lambda i:p2.fused_raw(fused_rows[i],xs[i]))
    score["full_score_order"],_=bench(all_rows,lambda i:(p2.fused_raw(fused_rows[i],xs[i]).argsort()[::-1]))

    print(json.dumps({
        "phase":"p15_feature_score_primitive_profile",
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "repeats":REPEATS,
        "feature_ns_per_call":feature,
        "score_ns_per_call":score,
        "note":"Primitive timings are isolated and non-additive; use them to choose the next tournament."
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
