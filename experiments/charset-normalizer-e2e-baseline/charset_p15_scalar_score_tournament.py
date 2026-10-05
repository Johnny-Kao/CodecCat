from __future__ import annotations

import argparse
import json
import math
import time

import joblib
import numpy as np

import charset_minimal_reranker_ab as rr
import charset_p2_multihop_kernel_tournament as p2
import charset_p3_allocation_lazy_tournament as p3
import charset_p8_context_metadata_tournament as p8
import charset_p14_calibrator_hotpath_tournament as p14

TIMING_REPEATS=25
REDUCE_IDX=np.asarray([32,127,128],dtype=np.intp)
AGG_MASK=np.zeros((2,256),dtype=np.int64)
AGG_MASK[0,32:127]=1
AGG_MASK[1,128:]=1

def features(data,mode):
    data=p2.hmt768(data)
    a=np.frombuffer(data,dtype=np.uint8)
    n=max(1,len(a))
    counts=np.bincount(a,minlength=256)
    out=np.empty(518,dtype=np.float32)
    out[:256]=counts
    out[:256]*=1.0/n
    if len(a)>=2:
        bins=a[:-1]+a[1:]
        bc=np.bincount(bins,minlength=256)
        out[256:512]=bc
        out[256:512]*=1.0/(len(a)-1)
    else:
        out[256:512]=0.0

    if len(a):
        inv=1.0/len(a)
        if mode=="baseline":
            printable_count=counts[32:127].sum()
            high_count=counts[128:].sum()
        elif mode=="reduceat":
            agg=np.add.reduceat(counts,REDUCE_IDX)
            printable_count=agg[0]
            high_count=agg[2]
        elif mode=="mask_matmul":
            agg=AGG_MASK@counts
            printable_count=agg[0]
            high_count=agg[1]
        else:
            raise ValueError(mode)
        high=float(high_count*inv)
        nul=float(counts[0]*inv)
        printable=float(printable_count*inv)
        lf=float(counts[10]*inv)
        cr=float(counts[13]*inv)
    else:
        high=nul=printable=lf=cr=0.0
    out[512:]=(math.log2(n+1)/16.0,high,nul,printable,lf,cr)
    return out

def raw_score(fused,x,mode):
    classes,w,b=fused
    xx=np.asarray(x,dtype=np.float64)
    if mode=="baseline":
        z=xx@w.T+b
    elif mode=="w_matmul":
        z=w@xx+b
    elif mode=="dot":
        z=np.dot(w,xx)+b
    elif mode=="einsum":
        z=np.einsum("ij,j->i",w,xx,optimize=False)+b
    else:
        raise ValueError(mode)
    if len(classes)==2 and z.shape==(1,):
        s=float(z[0])
        return np.asarray([-s,s],dtype=np.float64)
    return np.asarray(z,dtype=np.float64)

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

def build_ctx(s,fused,meta,feature_mode,score_mode):
    x=features(s["data"],feature_mode)
    raw=raw_score(fused,x,score_mode)
    order=raw.argsort()[::-1]
    vals=byte_values(s["data"])
    return p3.Ctx(
        data=s["data"],route=s["route"],
        rank=tuple(meta.classes_array[order]),
        sorted_scores=raw[order],raw_scores=raw,classes=meta.classes_tuple,
        has_utf8_bom=vals[0],strict_utf8=vals[1],ascii_only=vals[2],
        high_byte_ratio=vals[3],nul_ratio=vals[4],sample_len_4k=vals[5],
        replacement_rate=None,
    )

CANDIDATES={
    "p14_baseline":("baseline","baseline"),
    "feature_reduceat":("reduceat","baseline"),
    "feature_mask":("mask_matmul","baseline"),
    "score_w_matmul":("baseline","w_matmul"),
    "score_dot":("baseline","dot"),
    "score_einsum":("baseline","einsum"),
    "reduceat_w_matmul":("reduceat","w_matmul"),
    "reduceat_dot":("reduceat","dot"),
    "mask_w_matmul":("mask_matmul","w_matmul"),
    "mask_dot":("mask_matmul","dot"),
}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    assert state["schema_version"]==1
    assert sum(len(f["rows"]) for f in state["folds"])==418
    classes,families=state["classes"],state["families"]
    pooled={n:{"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}
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
        refs=[
            ds.run(build_ctx(s,fused[s["route"]],meta[s["route"]],"baseline","baseline"))
            for s in rows
        ]
        fs={n:{"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}
        for s,ref in zip(rows,refs):
            for name,(fm,sm) in CANDIDATES.items():
                out=ds.run(build_ctx(s,fused[s["route"]],meta[s["route"]],fm,sm))
                st=fs[name]
                st["n"]+=1
                st["hits"]+=int(out and out[0]==s["label"])
                st["mismatch"]+=int(list(out)!=list(ref))
        for name,(fm,sm) in CANDIDATES.items():
            if fs[name]["mismatch"]: continue
            sink=0; t0=time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    out=ds.run(build_ctx(s,fused[s["route"]],meta[s["route"]],fm,sm))
                    sink+=len(out)
            fs[name]["elapsed"]=time.perf_counter_ns()-t0
            fs[name]["calls"]=len(rows)*TIMING_REPEATS
            fs[name]["sink"]=sink
        fr={"fold":fd["fold"],"n":len(rows),"candidates":{}}
        for name,st in fs.items():
            fr["candidates"][name]={
                "top1":st["hits"]/max(1,st["n"]),
                "mismatch_n":st["mismatch"],
                "ns_per_call":st["elapsed"]/st["calls"] if st["calls"] else None,
            }
            for k in ("n","hits","mismatch","elapsed","calls"): pooled[name][k]+=st[k]
        folds.append(fr)
    out={}
    for name,st in pooled.items():
        ns=st["elapsed"]/st["calls"] if st["calls"] else None
        out[name]={"n":st["n"],"hits":st["hits"],"top1":st["hits"]/max(1,st["n"]),"mismatch_n":st["mismatch"],"ns_per_call":ns}
    base=out["p14_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p14"]=base/row["ns_per_call"]
            row["runtime_reduction_vs_p14"]=1-row["ns_per_call"]/base
    eligible=[(r["ns_per_call"],n) for n,r in out.items() if r["mismatch_n"]==0 and r["hits"]==365 and r["ns_per_call"] is not None]
    winner=min(eligible)[1] if eligible else None
    print(json.dumps({
        "phase":"p15_scalar_score_multihop_tournament",
        "canonical_target":{"n":418,"hits":365,"top1":365/418},
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "timing_repeats":TIMING_REPEATS,
        "pooled":out,"winner":winner,"folds":folds,
        "decision_rule":"Preserve exact P14 final ranking and 365/418; choose fastest same-run full-pipeline candidate."
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
