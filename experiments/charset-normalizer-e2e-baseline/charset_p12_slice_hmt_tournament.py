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
import charset_p7_specialist_microkernel_tournament as p7
import charset_p8_context_metadata_tournament as p8

TIMING_REPEATS = 27
UTF8_BOM = bytes((0xEF, 0xBB, 0xBF))


def hmt_array(data: bytes, mode: str):
    if mode == "bytes":
        return np.frombuffer(p2.hmt768(data), dtype=np.uint8)

    full = np.frombuffer(data, dtype=np.uint8)
    if len(full) <= 768:
        return full
    m = len(full) // 2

    if mode == "array_concat":
        return np.concatenate((full[:256], full[m-128:m+128], full[-256:]))
    if mode == "array_prealloc":
        out = np.empty(768, dtype=np.uint8)
        out[:256] = full[:256]
        out[256:512] = full[m-128:m+128]
        out[512:] = full[-256:]
        return out
    raise ValueError(mode)


def features_locked(data: bytes, hmt_mode: str):
    a = hmt_array(data, hmt_mode)
    n = max(1, len(a))
    counts = np.bincount(a, minlength=256)
    out = np.empty(518, dtype=np.float32)

    out[:256] = counts.astype(np.float32) / n

    if len(a) >= 2:
        bins = a[:-1] + a[1:]
        out[256:512] = np.bincount(bins, minlength=256).astype(np.float32) / (len(a) - 1)
    else:
        out[256:512] = 0.0

    if len(a):
        inv = 1.0 / len(a)
        high = float(counts[128:].sum() * inv)
        nul = float(counts[0] * inv)
        printable = float(counts[32:127].sum() * inv)
        lf = float(counts[10] * inv)
        cr = float(counts[13] * inv)
    else:
        high = nul = printable = lf = cr = 0.0

    out[512:] = (math.log2(n + 1) / 16.0, high, nul, printable, lf, cr)
    return out


def strict_from_sample(sample: bytes):
    try:
        sample.decode("utf-8", "strict")
        return True
    except UnicodeDecodeError:
        return False


def build_ctx(s, fused_model, meta, cfg):
    data = s["data"]
    x = features_locked(data, cfg["hmt"])
    raw = p2.fused_raw(fused_model, x)
    order = raw.argsort()[::-1]

    sample = data[:4096]
    arr = np.frombuffer(sample, dtype=np.uint8)

    strict = strict_from_sample(sample) if cfg["sample_decode"] else rr.strict_utf8(data)
    bom = sample.startswith(UTF8_BOM) if cfg["inline_bom"] else rr.has_utf8_bom(data)

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
        data=data, route=s["route"], rank=tuple(meta.classes_array[order]),
        sorted_scores=raw[order], raw_scores=raw, classes=meta.classes_tuple,
        has_utf8_bom=bom, strict_utf8=strict, ascii_only=ascii_flag,
        high_byte_ratio=high, nul_ratio=nul, sample_len_4k=len(arr),
        replacement_rate=None,
    )


CANDIDATES = {
    "p11_baseline": {"hmt": "bytes", "sample_decode": False, "inline_bom": False},
    "hop1_sample_decode": {"hmt": "bytes", "sample_decode": True, "inline_bom": False},
    "hop1_inline_bom": {"hmt": "bytes", "sample_decode": False, "inline_bom": True},
    "hop1_sample_checks": {"hmt": "bytes", "sample_decode": True, "inline_bom": True},
    "hop1_hmt_concat": {"hmt": "array_concat", "sample_decode": False, "inline_bom": False},
    "hop1_hmt_prealloc": {"hmt": "array_prealloc", "sample_decode": False, "inline_bom": False},
    "hop2_concat_checks": {"hmt": "array_concat", "sample_decode": True, "inline_bom": True},
    "hop2_prealloc_checks": {"hmt": "array_prealloc", "sample_decode": True, "inline_bom": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()
    state = joblib.load(args.state)
    assert state["schema_version"] == 1
    assert sum(len(f["rows"]) for f in state["folds"]) == 418

    classes, families = state["classes"], state["families"]
    pooled = {n: {"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}
    folds = []

    for fd in state["folds"]:
        models, rows = fd["models"], fd["rows"]
        fused = {r:p2.fuse_scaler_linear(mt) for r,mt in models.items()}
        meta = {r:p8.RouteMeta(mt) for r,mt in models.items()}
        ds = p7.P7Downstream(
            fd["cal"], fd["triad_cal"], fd["sig_cal"], fd["gb_cal"],
            classes, families, models, True, scalar_triad=True, inline_pairs=False
        )

        refs = [ds.run(build_ctx(s, fused[s["route"]], meta[s["route"]], CANDIDATES["p11_baseline"])) for s in rows]
        fs = {n: {"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}

        for s, ref in zip(rows, refs):
            for name,cfg in CANDIDATES.items():
                out = ds.run(build_ctx(s, fused[s["route"]], meta[s["route"]], cfg))
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
    base_ns=out["p11_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p11"]=base_ns/row["ns_per_call"]
            row["runtime_reduction_vs_p11"]=1-row["ns_per_call"]/base_ns
    eligible=[(r["ns_per_call"],n) for n,r in out.items()
              if r["mismatch_n"]==0 and r["hits"]==365 and r["ns_per_call"] is not None]
    winner=min(eligible)[1] if eligible else None
    print(json.dumps({
        "phase":"p12_slice_hmt_multihop_tournament",
        "canonical_target":{"n":418,"hits":365,"top1":365/418},
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "timing_repeats":TIMING_REPEATS,
        "pooled":out,"winner":winner,"folds":folds,
        "decision_rule":"Preserve exact P11 final ranking and 365/418; choose fastest same-run exact-equivalent candidate."
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
