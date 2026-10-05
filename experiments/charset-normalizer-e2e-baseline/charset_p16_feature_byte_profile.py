from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np

import charset_p2_multihop_kernel_tournament as p2
import charset_p15_scalar_score_tournament as p15

REPEATS=120

def bench(items,fn):
    n=len(items); sink=0
    t0=time.perf_counter_ns()
    for _ in range(REPEATS):
        for i in range(n):
            x=fn(i)
            if isinstance(x,np.ndarray): sink+=x.size
            elif isinstance(x,(tuple,list)): sink+=len(x)
            elif x is not None: sink+=1
    return (time.perf_counter_ns()-t0)/(REPEATS*n),sink

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    rows=[s for fd in state["folds"] for s in fd["rows"]]
    sampled=[p2.hmt768(s["data"]) for s in rows]
    arrs=[np.frombuffer(x,dtype=np.uint8) for x in sampled]
    counts=[np.bincount(a,minlength=256) for a in arrs]
    bins=[a[:-1]+a[1:] if len(a)>=2 else np.empty(0,dtype=np.uint8) for a in arrs]
    bcounts=[np.bincount(b,minlength=256) if len(b) else np.zeros(256,dtype=np.int64) for b in bins]
    outs=[np.empty(518,dtype=np.float32) for _ in rows]

    feature={}
    feature["unigram_bincount"],_=bench(rows,lambda i:np.bincount(arrs[i],minlength=256))
    feature["bigram_add"],_=bench(rows,lambda i:arrs[i][:-1]+arrs[i][1:] if len(arrs[i])>=2 else np.empty(0,dtype=np.uint8))
    feature["bigram_bincount"],_=bench(rows,lambda i:np.bincount(bins[i],minlength=256) if len(bins[i]) else np.zeros(256,dtype=np.int64))
    def norm_uni(i):
        out=outs[i]
        out[:256]=counts[i]
        out[:256]*=1.0/max(1,len(arrs[i]))
        return out[:256]
    def norm_bi(i):
        out=outs[i]
        if len(arrs[i])>=2:
            out[256:512]=bcounts[i]
            out[256:512]*=1.0/(len(arrs[i])-1)
        else:
            out[256:512]=0.0
        return out[256:512]
    feature["normalize_unigram"],_=bench(rows,norm_uni)
    feature["normalize_bigram"],_=bench(rows,norm_bi)
    feature["reduceat_stats"],_=bench(rows,lambda i:np.add.reduceat(counts[i],p15.REDUCE_IDX))
    feature["full_feature"],_=bench(rows,lambda i:p15.features(rows[i]["data"],"reduceat"))

    samples=[s["data"][:4096] for s in rows]
    orig=[np.frombuffer(x,dtype=np.uint8) for x in samples]
    byte={}
    def decode(i):
        try:
            samples[i].decode("utf-8",errors="strict")
            return True
        except UnicodeDecodeError:
            return False
    byte["strict_decode"],_=bench(rows,decode)
    byte["high_count"],_=bench(rows,lambda i:np.count_nonzero(orig[i]>=128))
    byte["nul_count"],_=bench(rows,lambda i:np.count_nonzero(orig[i]==0))
    byte["two_counts"],_=bench(rows,lambda i:(np.count_nonzero(orig[i]>=128),np.count_nonzero(orig[i]==0)))
    byte["bom"],_=bench(rows,lambda i:samples[i].startswith(b"\xef\xbb\xbf"))

    # data distribution to estimate safe ASCII fast-path opportunity
    ascii_n=0
    for a in orig:
        ascii_n+=int(len(a)==0 or np.count_nonzero(a>=128)==0)

    print(json.dumps({
        "phase":"p16_feature_byte_primitive_profile",
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "repeats":REPEATS,
        "feature_ns_per_call":feature,
        "byte_ns_per_call":byte,
        "ascii_samples":ascii_n,
        "n":len(rows),
        "ascii_rate":ascii_n/max(1,len(rows)),
        "note":"Use primitive costs and ASCII prevalence to decide whether a final P16 tournament is justified."
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
