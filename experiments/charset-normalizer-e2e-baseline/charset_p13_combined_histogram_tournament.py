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
import charset_p12_normalized_count_tournament as p12

TIMING_REPEATS = 25


def _finish_features(a, counts, bc):
    n = max(1, len(a))
    out = np.empty(518, dtype=np.float32)
    out[:256] = counts
    out[:256] *= 1.0 / n
    if len(a) >= 2:
        out[256:512] = bc
        out[256:512] *= 1.0 / (len(a) - 1)
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


def features_kernel(data: bytes, mode: str):
    data = p2.hmt768(data)
    a = np.frombuffer(data, dtype=np.uint8)
    m = len(a)

    if mode == "two_bincounts":
        counts = np.bincount(a, minlength=256)
        if m >= 2:
            bins = a[:-1] + a[1:]
            bc = np.bincount(bins, minlength=256)
        else:
            bc = np.zeros(256, dtype=np.int64)
        return _finish_features(a, counts, bc)

    if mode in ("combined_uint16", "combined_intp"):
        dtype = np.uint16 if mode == "combined_uint16" else np.intp
        idx = np.empty(m + max(0, m - 1), dtype=dtype)
        if m:
            idx[:m] = a
        if m >= 2:
            np.add(a[:-1], a[1:], out=idx[m:], casting="unsafe")
            idx[m:] += 256
        hist = np.bincount(idx, minlength=512)
        return _finish_features(a, hist[:256], hist[256:512])

    raise ValueError(mode)


def build_ctx(s, fused_model, meta, mode):
    x = features_kernel(s["data"], mode)
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
    "p12_baseline": "two_bincounts",
    "combined_uint16": "combined_uint16",
    "combined_intp": "combined_intp",
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

        refs = [
            ds.run(build_ctx(s, fused[s["route"]], meta[s["route"]], CANDIDATES["p12_baseline"]))
            for s in rows
        ]
        fs = {n: {"n": 0, "hits": 0, "mismatch": 0, "elapsed": 0, "calls": 0} for n in CANDIDATES}

        for s, ref in zip(rows, refs):
            for name, mode in CANDIDATES.items():
                out = ds.run(build_ctx(s, fused[s["route"]], meta[s["route"]], mode))
                st = fs[name]
                st["n"] += 1
                st["hits"] += int(out and out[0] == s["label"])
                st["mismatch"] += int(list(out) != list(ref))

        for name, mode in CANDIDATES.items():
            if fs[name]["mismatch"]:
                continue
            sink = 0
            t0 = time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    out = ds.run(build_ctx(s, fused[s["route"]], meta[s["route"]], mode))
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

    base_ns = out["p12_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p12"] = base_ns / row["ns_per_call"]
            row["runtime_reduction_vs_p12"] = 1 - row["ns_per_call"] / base_ns

    eligible = [
        (r["ns_per_call"], n)
        for n, r in out.items()
        if r["mismatch_n"] == 0 and r["hits"] == 365 and r["ns_per_call"] is not None
    ]
    winner = min(eligible)[1] if eligible else None

    print(json.dumps({
        "phase": "p13_combined_histogram_tournament",
        "canonical_target": {"n": 418, "hits": 365, "top1": 365 / 418},
        "cache_schema": state["schema_version"],
        "corpus_fingerprint": state["corpus_fingerprint"],
        "timing_repeats": TIMING_REPEATS,
        "pooled": out, "winner": winner, "folds": folds,
        "decision_rule": "Preserve exact P12 final ranking and 365/418; choose fastest same-run exact-equivalent candidate.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
