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

TIMING_REPEATS = 25


def features_kernel(data: bytes, bin_mode: str, divide_out: bool):
    data = p2.hmt768(data)
    a = np.frombuffer(data, dtype=np.uint8)
    n = max(1, len(a))
    counts = np.bincount(a, minlength=256)
    out = np.empty(518, dtype=np.float32)

    u = counts.astype(np.float32)
    if divide_out:
        np.divide(u, n, out=out[:256])
    else:
        out[:256] = u / n

    if len(a) >= 2:
        if bin_mode == "baseline":
            bins = ((a[:-1].astype(np.uint16) + a[1:].astype(np.uint16)) & 255).astype(np.int32)
        elif bin_mode == "uint8_op":
            bins = a[:-1] + a[1:]
        elif bin_mode == "uint8_add":
            bins = np.add(a[:-1], a[1:], dtype=np.uint8)
        else:
            raise ValueError(bin_mode)

        bc = np.bincount(bins, minlength=256).astype(np.float32)
        if divide_out:
            np.divide(bc, len(a) - 1, out=out[256:512])
        else:
            out[256:512] = bc / (len(a) - 1)
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


def build_ctx(s, fused_model, meta, cfg):
    x = features_kernel(s["data"], cfg["bin_mode"], cfg["divide_out"])
    raw = p2.fused_raw(fused_model, x)
    order = raw.argsort()[::-1]
    sample = s["data"][:4096]
    arr = np.frombuffer(sample, dtype=np.uint8)
    strict = rr.strict_utf8(s["data"])
    bom = rr.has_utf8_bom(s["data"])
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
    "p10_baseline": {"bin_mode": "baseline", "divide_out": False},
    "hop1_uint8_op": {"bin_mode": "uint8_op", "divide_out": False},
    "hop1_uint8_add": {"bin_mode": "uint8_add", "divide_out": False},
    "hop1_divide_out": {"bin_mode": "baseline", "divide_out": True},
    "hop2_uint8_op_divide": {"bin_mode": "uint8_op", "divide_out": True},
    "hop2_uint8_add_divide": {"bin_mode": "uint8_add", "divide_out": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()
    state = joblib.load(args.state)
    assert state["schema_version"] == 1
    assert sum(len(f["rows"]) for f in state["folds"]) == 418

    classes = state["classes"]
    families = state["families"]
    pooled = {n: {"n": 0, "hits": 0, "mismatch": 0, "elapsed": 0, "calls": 0} for n in CANDIDATES}
    folds = []

    for fd in state["folds"]:
        models = fd["models"]
        rows = fd["rows"]
        fused = {r: p2.fuse_scaler_linear(mt) for r, mt in models.items()}
        meta = {r: p8.RouteMeta(mt) for r, mt in models.items()}
        ds = p7.P7Downstream(
            fd["cal"], fd["triad_cal"], fd["sig_cal"], fd["gb_cal"],
            classes, families, models, True,
            scalar_triad=True, inline_pairs=False,
        )

        refs = []
        for s in rows:
            refs.append(ds.run(build_ctx(s, fused[s["route"]], meta[s["route"]], CANDIDATES["p10_baseline"])))

        fs = {n: {"n": 0, "hits": 0, "mismatch": 0, "elapsed": 0, "calls": 0} for n in CANDIDATES}

        for s, ref in zip(rows, refs):
            for name, cfg in CANDIDATES.items():
                out = ds.run(build_ctx(s, fused[s["route"]], meta[s["route"]], cfg))
                st = fs[name]
                st["n"] += 1
                st["hits"] += int(out and out[0] == s["label"])
                st["mismatch"] += int(list(out) != list(ref))

        for name, cfg in CANDIDATES.items():
            if fs[name]["mismatch"]:
                continue
            sink = 0
            t0 = time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    out = ds.run(build_ctx(s, fused[s["route"]], meta[s["route"]], cfg))
                    sink += len(out)
            fs[name]["elapsed"] = time.perf_counter_ns() - t0
            fs[name]["calls"] = len(rows) * TIMING_REPEATS
            fs[name]["sink"] = sink

        fr = {"fold": fd["fold"], "n": len(rows), "candidates": {}}
        for name, st in fs.items():
            fr["candidates"][name] = {
                "top1": st["hits"] / max(1, st["n"]),
                "mismatch_n": st["mismatch"],
                "ns_per_call": st["elapsed"] / st["calls"] if st["calls"] else None,
            }
            for k in ("n", "hits", "mismatch", "elapsed", "calls"):
                pooled[name][k] += st[k]
        folds.append(fr)

    out = {}
    for name, st in pooled.items():
        ns = st["elapsed"] / st["calls"] if st["calls"] else None
        out[name] = {
            "n": st["n"], "hits": st["hits"],
            "top1": st["hits"] / max(1, st["n"]),
            "mismatch_n": st["mismatch"], "ns_per_call": ns,
        }

    base_ns = out["p10_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p10"] = base_ns / row["ns_per_call"]
            row["runtime_reduction_vs_p10"] = 1 - row["ns_per_call"] / base_ns

    eligible = [
        (r["ns_per_call"], n)
        for n, r in out.items()
        if r["mismatch_n"] == 0 and r["hits"] == 365 and r["ns_per_call"] is not None
    ]
    winner = min(eligible)[1] if eligible else None

    print(json.dumps({
        "phase": "p11_feature_microkernel_multihop_tournament",
        "canonical_target": {"n": 418, "hits": 365, "top1": 365 / 418},
        "cache_schema": state["schema_version"],
        "corpus_fingerprint": state["corpus_fingerprint"],
        "timing_repeats": TIMING_REPEATS,
        "pooled": out, "winner": winner, "folds": folds,
        "decision_rule": "Preserve exact P10 final ranking and 365/418; choose fastest same-run exact-equivalent candidate.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
