from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np

import charset_p2_multihop_kernel_tournament as p2
import charset_p3_allocation_lazy_tournament as p3
import charset_p8_context_metadata_tournament as p8
import charset_p14_calibrator_hotpath_tournament as p14
import charset_p15_scalar_score_tournament as p15

REPEATS=60

def byte_values(data):
    sample=data[:4096]
    arr=np.frombuffer(sample,dtype=np.uint8)
    try:
        sample.decode("utf-8",errors="strict")
        strict=True
    except UnicodeDecodeError:
        strict=False
    bom=sample.startswith(b"\xef\xbb\xbf")
    if len(arr):
        high_count=int(np.count_nonzero(arr>=128))
        nul_count=int(np.count_nonzero(arr==0))
        high=high_count/len(arr)
        nul=nul_count/len(arr)
        ascii_flag=(high_count==0)
    else:
        high=nul=0.0
        ascii_flag=True
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

def full_one(s,fused,meta,ds):
    x=p15.features(s["data"],"reduceat")
    raw=p15.raw_score(fused,x,"dot")
    order=raw.argsort()[::-1]
    vals=byte_values(s["data"])
    return ds.run(make_ctx(s,raw,order,meta,vals))

def bench(fn,n):
    sink=0
    t0=time.perf_counter_ns()
    for _ in range(REPEATS):
        for i in range(n):
            x=fn(i)
            if isinstance(x,(tuple,list,np.ndarray)):
                sink+=len(x)
            elif x is not None:
                sink+=1
    return (time.perf_counter_ns()-t0)/(REPEATS*n),sink

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",required=True)
    args=ap.parse_args()
    state=joblib.load(args.state)
    assert state["schema_version"]==1
    assert sum(len(f["rows"]) for f in state["folds"])==418

    classes,families=state["classes"],state["families"]
    totals={k:[0.0,0] for k in ("feature","score","order","byte","ctx","downstream","full")}
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

        xs=[p15.features(s["data"],"reduceat") for s in rows]
        raws=[p15.raw_score(fused[s["route"]],x,"dot") for s,x in zip(rows,xs)]
        orders=[raw.argsort()[::-1] for raw in raws]
        vals=[byte_values(s["data"]) for s in rows]
        ctxs=[make_ctx(s,raw,order,meta[s["route"]],v) for s,raw,order,v in zip(rows,raws,orders,vals)]

        n=len(rows); result={}
        result["feature"],_=bench(lambda i:p15.features(rows[i]["data"],"reduceat"),n)
        result["score"],_=bench(lambda i:p15.raw_score(fused[rows[i]["route"]],xs[i],"dot"),n)
        result["order"],_=bench(lambda i:raws[i].argsort()[::-1],n)
        result["byte"],_=bench(lambda i:byte_values(rows[i]["data"]),n)
        result["ctx"],_=bench(lambda i:make_ctx(rows[i],raws[i],orders[i],meta[rows[i]["route"]],vals[i]),n)
        result["downstream"],_=bench(lambda i:ds.run(ctxs[i]),n)
        result["full"],_=bench(lambda i:full_one(rows[i],fused[rows[i]["route"]],meta[rows[i]["route"]],ds),n)

        for k,v in result.items():
            totals[k][0]+=v*n; totals[k][1]+=n
        folds.append({"fold":fd["fold"],"n":n,"ns_per_call":result})

    pooled={k:t/n for k,(t,n) in totals.items()}
    full=pooled["full"]
    shares={k:pooled[k]/full for k in ("feature","score","order","byte","ctx","downstream")}

    print(json.dumps({
        "phase":"p16_locked_p15_residual_profile",
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "repeats":REPEATS,
        "pooled_ns_per_call":pooled,
        "isolated_share_vs_full":shares,
        "folds":folds,
        "note":"Directional isolated timings are non-additive; use to decide whether P16 has a worthwhile final optimization family."
    },indent=2,sort_keys=True))

if __name__=="__main__":
    main()
