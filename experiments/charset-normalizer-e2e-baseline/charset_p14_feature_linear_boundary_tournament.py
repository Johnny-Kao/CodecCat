from __future__ import annotations

import argparse
import json
import math
import time

import joblib
import numpy as np

import charset_p2_multihop_kernel_tournament as p2
import charset_p3_allocation_lazy_tournament as p3
import charset_p7_specialist_microkernel_tournament as p7
import charset_p8_context_metadata_tournament as p8
import charset_p12_slice_hmt_tournament as p12

TIMING_REPEATS = 27


def features_f64_exact(data: bytes):
    # Preserve the locked float32 feature values exactly, but store the final
    # vector in float64 so fused_raw no longer needs a 518-element cast/copy.
    a = np.frombuffer(p2.hmt768(data), dtype=np.uint8)
    n = max(1, len(a))
    counts = np.bincount(a, minlength=256)
    out = np.empty(518, dtype=np.float64)

    out[:256] = counts.astype(np.float32) / n

    if len(a) >= 2:
        bins = a[:-1] + a[1:]
        out[256:512] = (
            np.bincount(bins, minlength=256).astype(np.float32) / (len(a) - 1)
        )
    else:
        out[256:512] = 0.0

    if len(a):
        inv = 1.0 / len(a)
        vals = (
            math.log2(n + 1) / 16.0,
            float(counts[128:].sum() * inv),
            float(counts[0] * inv),
            float(counts[32:127].sum() * inv),
            float(counts[10] * inv),
            float(counts[13] * inv),
        )
    else:
        vals = (math.log2(n + 1) / 16.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Match assignment into the locked float32 output element-by-element.
    for i, v in enumerate(vals):
        out[512 + i] = np.float32(v)
    return out


def raw_gemv(fused, x):
    classes, w, b = fused
    x64 = np.asarray(x, dtype=np.float64)
    z = w @ x64 + b
    if len(classes) == 2 and z.shape == (1,):
        s = float(z[0])
        return np.asarray([-s, s], dtype=np.float64)
    return np.asarray(z, dtype=np.float64)


def build_ctx(s, fused_model, meta, cfg):
    x = features_f64_exact(s["data"]) if cfg["f64"] else p12.features_locked(s["data"], "bytes")
    raw = raw_gemv(fused_model, x) if cfg["gemv"] else p2.fused_raw(fused_model, x)
    order = raw.argsort()[::-1]

    sample = s["data"][:4096]
    arr = np.frombuffer(sample, dtype=np.uint8)
    strict = p12.strict_from_sample(sample)
    bom = sample.startswith(p12.UTF8_BOM)

    if len(arr):
        high_count = int(np.count_nonzero(arr >= 128))
        nul_count = int(np.count_nonzero(arr == 0))
        high = high_count / len(arr)
        nul = nul_count / len(arr)
        ascii_flag = high_count == 0
    else:
        high = nul = 0.0
        ascii_flag = True

    return p3.Ctx(
        data=s["data"], route=s["route"],
        rank=tuple(meta.classes_array[order]),
        sorted_scores=raw[order], raw_scores=raw, classes=meta.classes_tuple,
        has_utf8_bom=bom, strict_utf8=strict, ascii_only=ascii_flag,
        high_byte_ratio=high, nul_ratio=nul, sample_len_4k=len(arr),
        replacement_rate=None,
    )


CANDIDATES = {
    "p12_baseline": {"f64": False, "gemv": False},
    "hop1_feature_f64": {"f64": True, "gemv": False},
    "hop1_gemv": {"f64": False, "gemv": True},
    "hop2_f64_gemv": {"f64": True, "gemv": True},
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
        models,rows=fd["models"],fd["rows"]
        fused={r:p2.fuse_scaler_linear(mt) for r,mt in models.items()}
        meta={r:p8.RouteMeta(mt) for r,mt in models.items()}
        ds=p7.P7Downstream(
            fd["cal"],fd["triad_cal"],fd["sig_cal"],fd["gb_cal"],
            classes,families,models,True,scalar_triad=True,inline_pairs=False
        )

        refs=[ds.run(build_ctx(s,fused[s["route"]],meta[s["route"]],CANDIDATES["p12_baseline"])) for s in rows]
        fs={n:{"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}

        for s,ref in zip(rows,refs):
            for name,cfg in CANDIDATES.items():
                out=ds.run(build_ctx(s,fused[s["route"]],meta[s["route"]],cfg))
                st=fs[name]; st["n"]+=1
                st["hits"]+=int(out and out[0]==s["label"])
                st["mismatch"]+=int(list(out)!=list(ref))

        for name,cfg in CANDIDATES.items():
            if fs[name]["mismatch"]: continue
            sink=0; t0=time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    out=ds.run(build_ctx(s,fused[s["route"]],meta[s["route"]],cfg))
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
        out[name]={"n":st["n"],"hits":st["hits"],"top1":st["hits"]/max(1,st["n"]),
                   "mismatch_n":st["mismatch"],"ns_per_call":ns}
    base_ns=out["p12_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p12"]=base_ns/row["ns_per_call"]
            row["runtime_reduction_vs_p12"]=1-row["ns_per_call"]/base_ns
    eligible=[(r["ns_per_call"],n) for n,r in out.items()
              if r["mismatch_n"]==0 and r["hits"]==365 and r["ns_per_call"] is not None]
    winner=min(eligible)[1] if eligible else None
    print(json.dumps({
        "phase":"p14_feature_linear_boundary_tournament",
        "canonical_target":{"n":418,"hits":365,"top1":365/418},
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "timing_repeats":TIMING_REPEATS,
        "pooled":out,"winner":winner,"folds":folds,
        "decision_rule":"Preserve exact P12 final ranking and 365/418; choose fastest same-run exact-equivalent candidate."
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
