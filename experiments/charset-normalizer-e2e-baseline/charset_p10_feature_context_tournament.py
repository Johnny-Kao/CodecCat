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

TIMING_REPEATS = 23


def features_no_concat(data: bytes):
    data = p2.hmt768(data)
    a = np.frombuffer(data, dtype=np.uint8)
    n = max(1, len(a))
    counts = np.bincount(a, minlength=256)

    out = np.empty(518, dtype=np.float32)
    out[:256] = counts.astype(np.float32) / n

    if len(a) >= 2:
        bins = ((a[:-1].astype(np.uint16) + a[1:].astype(np.uint16)) & 255).astype(np.int32)
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

    out[512:] = (
        math.log2(n + 1) / 16.0,
        high,
        nul,
        printable,
        lf,
        cr,
    )
    return out


def features_with_short_stats(data: bytes, no_concat=False):
    # When the original input is <=768 bytes, hmt768 is identity and the
    # unigram counts are exactly the byte statistics needed by context setup.
    source = p2.hmt768(data)
    a = np.frombuffer(source, dtype=np.uint8)
    n = max(1, len(a))
    counts = np.bincount(a, minlength=256)

    if no_concat:
        out = np.empty(518, dtype=np.float32)
        out[:256] = counts.astype(np.float32) / n
        if len(a) >= 2:
            bins = ((a[:-1].astype(np.uint16) + a[1:].astype(np.uint16)) & 255).astype(np.int32)
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
        out[512:] = (
            math.log2(n + 1) / 16.0,
            high,
            nul,
            printable,
            lf,
            cr,
        )
        x = out
    else:
        unigram = counts.astype(np.float32) / n
        bigram = np.zeros(256, dtype=np.float32)
        if len(a) >= 2:
            bins = ((a[:-1].astype(np.uint16) + a[1:].astype(np.uint16)) & 255).astype(np.int32)
            bigram = np.bincount(bins, minlength=256).astype(np.float32) / (len(a) - 1)
        if len(a):
            inv = 1.0 / len(a)
            high = float(counts[128:].sum() * inv)
            nul = float(counts[0] * inv)
            printable = float(counts[32:127].sum() * inv)
            lf = float(counts[10] * inv)
            cr = float(counts[13] * inv)
        else:
            high = nul = printable = lf = cr = 0.0
        scalars = np.array(
            [math.log2(n + 1) / 16.0, high, nul, printable, lf, cr],
            dtype=np.float32,
        )
        x = np.concatenate([unigram, bigram, scalars])

    return x, counts if len(data) <= 768 else None


def build_ctx(
    model_tuple,
    s,
    fused_model,
    meta,
    *,
    nul_bytes_count=False,
    short_ascii_skip=False,
    feature_no_concat=False,
    reuse_short_stats=False,
):
    data = s["data"]

    if reuse_short_stats:
        x, reusable_counts = features_with_short_stats(data, no_concat=feature_no_concat)
    else:
        x = features_no_concat(data) if feature_no_concat else p2.features_combined(data)
        reusable_counts = None

    raw = p2.fused_raw(fused_model, x)
    order = raw.argsort()[::-1]

    sample = data[:4096]
    n4k = len(sample)

    if reusable_counts is not None:
        high_count = int(reusable_counts[128:].sum())
        nul_count = int(reusable_counts[0])
        high = high_count / n4k if n4k else 0.0
        nul = nul_count / n4k if n4k else 0.0
        ascii_flag = high_count == 0
    else:
        arr = np.frombuffer(sample, dtype=np.uint8)
        if len(arr):
            high_count = int(np.count_nonzero(arr >= 128))
            if nul_bytes_count:
                nul_count = sample.count(b"\x00")
            else:
                nul_count = int(np.count_nonzero(arr == 0))
            high = high_count / len(arr)
            nul = nul_count / len(arr)
            ascii_flag = high_count == 0
        else:
            high = nul = 0.0
            ascii_flag = True

    # Exact fast path only when the first-4k statistics cover the entire input.
    if short_ascii_skip and len(data) <= 4096 and ascii_flag:
        strict = True
        bom = False
    else:
        strict = rr.strict_utf8(data)
        bom = rr.has_utf8_bom(data)

    return p3.Ctx(
        data=data,
        route=s["route"],
        rank=tuple(meta.classes_array[order]),
        sorted_scores=raw[order],
        raw_scores=raw,
        classes=meta.classes_tuple,
        has_utf8_bom=bom,
        strict_utf8=strict,
        ascii_only=ascii_flag,
        high_byte_ratio=high,
        nul_ratio=nul,
        sample_len_4k=n4k,
        replacement_rate=None,
    )


CANDIDATES = {
    "p9_baseline": dict(nul=False, ascii=False, feat=False, reuse=False),
    "hop1_nul_bytes_count": dict(nul=True, ascii=False, feat=False, reuse=False),
    "hop1_short_ascii_skip": dict(nul=False, ascii=True, feat=False, reuse=False),
    "hop1_feature_no_concat": dict(nul=False, ascii=False, feat=True, reuse=False),
    "hop1_reuse_short_stats": dict(nul=False, ascii=False, feat=False, reuse=True),
    "hop2_nul_ascii": dict(nul=True, ascii=True, feat=False, reuse=False),
    "hop2_feature_reuse": dict(nul=False, ascii=False, feat=True, reuse=True),
    "hop3_all": dict(nul=True, ascii=True, feat=True, reuse=True),
}


def make_ctx(cfg, mt, s, fused_model, meta):
    return build_ctx(
        mt,
        s,
        fused_model,
        meta,
        nul_bytes_count=cfg["nul"],
        short_ascii_skip=cfg["ascii"],
        feature_no_concat=cfg["feat"],
        reuse_short_stats=cfg["reuse"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()

    state = joblib.load(args.state)
    assert state["schema_version"] == 1
    assert sum(len(f["rows"]) for f in state["folds"]) == 418

    classes = state["classes"]
    families = state["families"]
    pooled = {
        n: {"n": 0, "hits": 0, "mismatch": 0, "elapsed": 0, "calls": 0}
        for n in CANDIDATES
    }
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
            mt = models[s["route"]]
            refs.append(
                ds.run(
                    build_ctx(
                        mt, s, fused[s["route"]], meta[s["route"]]
                    )
                )
            )

        fs = {
            n: {"n": 0, "hits": 0, "mismatch": 0, "elapsed": 0, "calls": 0}
            for n in CANDIDATES
        }

        for s, ref in zip(rows, refs):
            mt = models[s["route"]]
            fm = fused[s["route"]]
            rm = meta[s["route"]]
            for name, cfg in CANDIDATES.items():
                out = ds.run(make_ctx(cfg, mt, s, fm, rm))
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
                    mt = models[s["route"]]
                    out = ds.run(
                        make_ctx(
                            cfg,
                            mt,
                            s,
                            fused[s["route"]],
                            meta[s["route"]],
                        )
                    )
                    sink += len(out)
            dt = time.perf_counter_ns() - t0
            fs[name]["elapsed"] = dt
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
            "n": st["n"],
            "hits": st["hits"],
            "top1": st["hits"] / max(1, st["n"]),
            "mismatch_n": st["mismatch"],
            "ns_per_call": ns,
        }

    base_ns = out["p9_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p9"] = base_ns / row["ns_per_call"]
            row["runtime_reduction_vs_p9"] = 1 - row["ns_per_call"] / base_ns

    eligible = [
        (r["ns_per_call"], n)
        for n, r in out.items()
        if r["mismatch_n"] == 0
        and r["hits"] == 365
        and r["ns_per_call"] is not None
    ]
    winner = min(eligible)[1] if eligible else None

    print(json.dumps({
        "phase": "p10_feature_context_multihop_tournament",
        "canonical_target": {"n": 418, "hits": 365, "top1": 365 / 418},
        "cache_schema": state["schema_version"],
        "corpus_fingerprint": state["corpus_fingerprint"],
        "timing_repeats": TIMING_REPEATS,
        "pooled": out,
        "winner": winner,
        "folds": folds,
        "decision_rule": (
            "Preserve exact P9 final ranking and 365/418; "
            "choose fastest same-run exact-equivalent candidate."
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
