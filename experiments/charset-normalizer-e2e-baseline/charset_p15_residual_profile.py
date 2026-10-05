from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np

import charset_minimal_reranker_ab as rr
import charset_p2_multihop_kernel_tournament as p2
import charset_p3_allocation_lazy_tournament as p3
import charset_p8_context_metadata_tournament as p8
import charset_p12_normalized_count_tournament as p12
import charset_p14_calibrator_hotpath_tournament as p14

REPEATS=50

def byte_values(data):
    sample=data[:4096]
    arr=np.frombuffer(sample,dtype=np.uint8)
    strict=rr.strict_utf8(data)
    bom=rr.has_utf8_bom(data)
    if len(arr):
        high_count=int(np.count_nonzero(arr>=128))
        nul_count=int(np.count_nonzero(arr==0))
        high=high_count/len(arr); nul=nul_count/len(arr); ascii_flag=(high_count==0)
    else:
        high=nul=0.0; ascii_flag=True
    return bom,strict,ascii_flag,high,nul,len(arr)

def make_ctx(s,raw,order,meta,vals):
    return p3.Ctx(
        data=s["data"],route=s["route"],
        rank=tuple(meta.classes_array[order]),
        sorted_scores=raw[order],raw_scores=raw,classes=meta.classes_tuple,
        has_utf8_bom=vals[0],strict_utf8=vals[1],ascii_only=vals[2],
        high_byte_ratio=vals[3],nul_ratio=vals[4],sample_len_4k=vals[5],
        replacement_rate=None,
    )

def full_one(s,fused_model,meta,ds):
    x=p12.features_kernel(s["data"],"assign_multiply","assign_multiply")
    raw=p2.fused_raw(fused_model,x)
    order=raw.argsort()[::-1]
    vals=byte_values(s["data"])
    return ds.run(make_ctx(s,raw,order,meta,vals))

def bench(fn,n):
    sink=0
    t0=time.perf_counter_ns()
    for _ in range(REPEATS):
        for i in range(n):
            x=fn(i)
            sink += len(x) if isinstance(x,(tuple,list,np.ndarray)) else int(x is not None)
    return (time.perf_counter_ns()-t0)/(REPEATS*n),sink

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    assert state["schema_version"]==1
    assert sum(len(f["rows"]) for f in state["folds"])==418
    classes,families=state["classes"],state["families"]
    totals={k:[0.0,0] for k in ("feature","score_order","byte","ctx","downstream","full")}
    folds=[]
    for fd in state["folds"]:
        rows=fd["rows"]; models=fd["models"]
        fused={r:p2.fuse_scaler_linear(mt) for r,mt in models.items()}
        meta={r:p8.RouteMeta(mt) for r,mt in models.items()}
        ds=p14.CalDownstream(
            fd["cal"],fd["triad_cal"],fd["sig_cal"],fd["gb_cal"],
            classes,families,models,True,
            scalar_triad=True,inline_pairs=False,mode="combined",
        )
        xs=[p12.features_kernel(s["data"],"assign_multiply","assign_multiply") for s in rows]
        raws=[p2.fused_raw(fused[s["route"]],x) for s,x in zip(rows,xs)]
        orders=[raw.argsort()[::-1] for raw in raws]
        vals=[byte_values(s["data"]) for s in rows]
        ctxs=[make_ctx(s,raw,order,meta[s["route"]],v) for s,raw,order,v in zip(rows,raws,orders,vals)]
        n=len(rows); result={}
        result["feature"],_=bench(lambda i:p12.features_kernel(rows[i]["data"],"assign_multiply","assign_multiply"),n)
        result["score_order"],_=bench(lambda i:(lambda r:(r,r.argsort()[::-1]))(p2.fused_raw(fused[rows[i]["route"]],xs[i])),n)
        result["byte"],_=bench(lambda i:byte_values(rows[i]["data"]),n)
        result["ctx"],_=bench(lambda i:make_ctx(rows[i],raws[i],orders[i],meta[rows[i]["route"]],vals[i]),n)
        result["downstream"],_=bench(lambda i:ds.run(ctxs[i]),n)
        result["full"],_=bench(lambda i:full_one(rows[i],fused[rows[i]["route"]],meta[rows[i]["route"]],ds),n)
        for k,v in result.items():
            totals[k][0]+=v*n; totals[k][1]+=n
        folds.append({"fold":fd["fold"],"n":n,"ns_per_call":result})
    pooled={k:t/n for k,(t,n) in totals.items()}
    full=pooled["full"]
    shares={k:pooled[k]/full for k in ("feature","score_order","byte","ctx","downstream")}
    print(json.dumps({
        "phase":"p15_locked_p14_residual_profile",
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "repeats":REPEATS,
        "pooled_ns_per_call":pooled,
        "isolated_share_vs_full":shares,
        "folds":folds,
        "note":"Directional isolated timings are not additive; use only to select P15 optimization family."
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
