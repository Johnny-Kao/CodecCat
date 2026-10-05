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

TIMING_REPEATS=35
DELETE_ASCII=bytes(range(128))

def byte_values(data,mode):
    sample=data[:4096]
    arr=np.frombuffer(sample,dtype=np.uint8)
    try:
        sample.decode("utf-8",errors="strict")
        strict=True
    except UnicodeDecodeError:
        strict=False
    bom=sample.startswith(b"\xef\xbb\xbf")

    if len(arr):
        if mode=="baseline":
            high_count=int(np.count_nonzero(arr>=128))
            nul_count=int(np.count_nonzero(arr==0))
        elif mode=="bytes_nul":
            high_count=int(np.count_nonzero(arr>=128))
            nul_count=sample.count(b"\x00")
        elif mode=="translate_high":
            high_count=len(sample.translate(None,DELETE_ASCII))
            nul_count=int(np.count_nonzero(arr==0))
        elif mode=="bytes_both":
            high_count=len(sample.translate(None,DELETE_ASCII))
            nul_count=sample.count(b"\x00")
        elif mode=="isascii_nul":
            # isascii can only replace the boolean flag; high ratio still requires exact count.
            high_count=int(np.count_nonzero(arr>=128))
            nul_count=sample.count(b"\x00")
        else:
            raise ValueError(mode)
        high=high_count/len(arr)
        nul=nul_count/len(arr)
        ascii_flag=sample.isascii() if mode=="isascii_nul" else (high_count==0)
    else:
        high=nul=0.0
        ascii_flag=True
    return bom,strict,ascii_flag,high,nul,len(arr)

def build_ctx(s,fused,meta,byte_mode):
    x=p15.features(s["data"],"reduceat")
    raw=p15.raw_score(fused,x,"dot")
    order=raw.argsort()[::-1]
    vals=byte_values(s["data"],byte_mode)
    return p3.Ctx(
        data=s["data"],route=s["route"],
        rank=tuple(meta.classes_array[order]),
        sorted_scores=raw[order],raw_scores=raw,classes=meta.classes_tuple,
        has_utf8_bom=vals[0],strict_utf8=vals[1],ascii_only=vals[2],
        high_byte_ratio=vals[3],nul_ratio=vals[4],sample_len_4k=vals[5],
        replacement_rate=None,
    )

CANDIDATES={
    "p15_baseline":"baseline",
    "bytes_nul":"bytes_nul",
    "translate_high":"translate_high",
    "bytes_both":"bytes_both",
    "isascii_nul":"isascii_nul",
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
        refs=[ds.run(build_ctx(s,fused[s["route"]],meta[s["route"]],"baseline")) for s in rows]
        fs={n:{"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}

        for s,ref in zip(rows,refs):
            for name,mode in CANDIDATES.items():
                out=ds.run(build_ctx(s,fused[s["route"]],meta[s["route"]],mode))
                st=fs[name]
                st["n"]+=1
                st["hits"]+=int(out and out[0]==s["label"])
                st["mismatch"]+=int(list(out)!=list(ref))

        for name,mode in CANDIDATES.items():
            if fs[name]["mismatch"]: continue
            sink=0; t0=time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    out=ds.run(build_ctx(s,fused[s["route"]],meta[s["route"]],mode))
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

    base=out["p15_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p15"]=base/row["ns_per_call"]
            row["runtime_reduction_vs_p15"]=1-row["ns_per_call"]/base

    eligible=[(r["ns_per_call"],n) for n,r in out.items() if r["mismatch_n"]==0 and r["hits"]==365 and r["ns_per_call"] is not None]
    winner=min(eligible)[1] if eligible else None

    print(json.dumps({
        "phase":"p16_final_byte_path_tournament",
        "canonical_target":{"n":418,"hits":365,"top1":365/418},
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "timing_repeats":TIMING_REPEATS,
        "pooled":out,"winner":winner,"folds":folds,
        "convergence_rule":"If best exact-equivalent full-pipeline improvement is <1% or fold-unstable, stop performance optimization after P16.",
        "decision_rule":"Preserve exact P15 final ranking and 365/418; accept only a material same-run full-pipeline win."
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
