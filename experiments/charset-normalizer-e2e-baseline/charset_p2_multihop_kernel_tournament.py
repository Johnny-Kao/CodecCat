from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter

import numpy as np

import charset_external_scorer_learning_curve as lc
import charset_external_scorer_domain_shift_ab as base
import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_canonical_guarded_final_validation as canon
import charset_minimal_reranker_ab as rr
import charset_p1_cached_inference_context_ab as p1

OUTER_FOLDS = 4
TRAIN_N = 300
TIMING_REPEATS = 7


def hmt768(data: bytes) -> bytes:
    if len(data) <= 768:
        return data
    m = len(data) // 2
    return data[:256] + data[m-128:m+128] + data[-256:]


def features_baseline(data: bytes) -> np.ndarray:
    return base.scorer_features(data)


def _feature_core(data: bytes, scalar_reuse: bool, simplified_bigram: bool) -> np.ndarray:
    data = hmt768(data)
    a = np.frombuffer(data, dtype=np.uint8)
    n = max(1, len(a))

    counts = np.bincount(a, minlength=256)
    unigram = counts.astype(np.float32) / n

    bigram = np.zeros(256, dtype=np.float32)
    if len(a) >= 2:
        if simplified_bigram:
            # Exact algebraic identity modulo 256:
            # (a * 257 + b) & 255 == (a + b) & 255
            bins = (
                (a[:-1].astype(np.uint16) + a[1:].astype(np.uint16)) & 255
            ).astype(np.int32)
        else:
            bins = (
                (a[:-1].astype(np.uint16) * 257 + a[1:].astype(np.uint16)) & 255
            ).astype(np.int32)
        bigram = np.bincount(bins, minlength=256).astype(np.float32) / (len(a) - 1)

    if scalar_reuse and len(a):
        inv = 1.0 / len(a)
        high = float(counts[128:].sum() * inv)
        nul = float(counts[0] * inv)
        printable = float(counts[32:127].sum() * inv)
        lf = float(counts[10] * inv)
        cr = float(counts[13] * inv)
    else:
        high = float(np.mean(a >= 128)) if len(a) else 0.0
        nul = float(np.mean(a == 0)) if len(a) else 0.0
        printable = float(np.mean((a >= 32) & (a <= 126))) if len(a) else 0.0
        lf = float(np.mean(a == 10)) if len(a) else 0.0
        cr = float(np.mean(a == 13)) if len(a) else 0.0

    scalars = np.array([
        math.log2(n + 1) / 16.0,
        high,
        nul,
        printable,
        lf,
        cr,
    ], dtype=np.float32)
    return np.concatenate([unigram, bigram, scalars])


def features_scalar_reuse(data: bytes) -> np.ndarray:
    return _feature_core(data, True, False)


def features_bigram_simplified(data: bytes) -> np.ndarray:
    return _feature_core(data, False, True)


def features_combined(data: bytes) -> np.ndarray:
    return _feature_core(data, True, True)


FEATURES = {
    "baseline": features_baseline,
    "scalar_reuse": features_scalar_reuse,
    "bigram_simplified": features_bigram_simplified,
    "combined": features_combined,
}


def score_standard(model_tuple, x):
    scaler, model = model_tuple
    scores = model.decision_function(scaler.transform(x[None, :]))
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores[0]


def fused_params(model_tuple):
    scaler, model = model_tuple
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    w = coef / scale[None, :]
    b = intercept - (coef * (mean / scale)[None, :]).sum(axis=1)
    return w, b


def score_fused(model_tuple, x, fused):
    _, model = model_tuple
    w, b = fused
    z = np.asarray(x, dtype=np.float64) @ w.T + b
    if len(model.classes_) == 2 and np.ndim(z) == 1 and z.shape[0] == 1:
        s = float(z[0])
        return np.asarray([-s, s], dtype=np.float64)
    return np.asarray(z, dtype=np.float64)


def byte_values(data):
    sample = data[:4096]
    arr = np.frombuffer(sample, dtype=np.uint8)
    return (
        rr.strict_utf8(data),
        rr.has_utf8_bom(data),
        rr.ascii_only(data),
        float(np.mean(arr >= 128)) if len(arr) else 0.0,
        float(np.mean(arr == 0)) if len(arr) else 0.0,
        len(arr),
        canon.replacement_rate(data),
    )


def build_ctx_from_raw(model_tuple, s, raw):
    _, model = model_tuple
    order = np.argsort(raw)[::-1]
    vals = byte_values(s["data"])
    return p1.InferenceContext(
        data=s["data"],
        route=s["route"],
        rank=tuple(model.classes_[order]),
        sorted_scores=np.asarray(raw)[order],
        score_map={c: float(v) for c, v in zip(model.classes_, raw)},
        has_utf8_bom=vals[1],
        strict_utf8=vals[0],
        ascii_only=vals[2],
        high_byte_ratio=vals[3],
        nul_ratio=vals[4],
        sample_len_4k=vals[5],
        replacement_rate=vals[6],
    )


def downstream(cal, triad_cal, sig_cal, gb_cal, ctx, classes, families):
    hybrid = p1.cached_hybrid(cal, ctx, classes, families)
    baseline = p1.cached_gate(triad_cal, ctx, hybrid)
    sig = p1.cached_apply_pair(sig_cal, p1.SIG_PAIR, ctx, baseline)
    out = p1.cached_apply_pair(gb_cal, p1.GB_PAIR, ctx, sig)
    if (
        out and sig and out[0] != sig[0]
        and sig[0] == "gb18030" and out[0] == "utf-8"
        and ctx.replacement_rate > p1.RATE_THRESHOLD
    ):
        out = sig
    return out


def main():
    external, _, stats = base.collect_external()
    legacy_X, legacy_y, legacy_b = base.load_legacy_training()
    classes, families = calmod.build_vocab(legacy_y)
    paths = sorted(set(s["warc_path"] for s in external))
    fold_assign = {
        p: int.from_bytes(
            hashlib.sha256(("learning-curve:" + p).encode()).digest()[:8], "big"
        ) % OUTER_FOLDS
        for p in paths
    }

    candidate_names = [
        "baseline+standard",
        "scalar_reuse+standard",
        "bigram_simplified+standard",
        "combined+standard",
        "baseline+fused",
        "scalar_reuse+fused",
        "bigram_simplified+fused",
        "combined+fused",
    ]

    agg = {name: Counter() for name in candidate_names}
    fold_rows = []

    for fold in range(OUTER_FOLDS):
        train_pool = [s for s in external if fold_assign[s["warc_path"]] != fold]
        test_rows = [s for s in external if fold_assign[s["warc_path"]] == fold]
        ext_train = lc.deterministic_nested_subset(train_pool, TRAIN_N)

        models = lc.fit_route_models(ext_train, legacy_X, legacy_y, legacy_b)
        cal = calmod.crossfit_calibration_rows(train_pool, legacy_X, legacy_y, legacy_b, classes, families)
        triad_cal = tri.crossfit_triad_rows(train_pool, legacy_X, legacy_y, legacy_b, classes, families)
        sig_cal = canon.fit_one(canon.SIG_PAIR, train_pool, legacy_X, legacy_y, legacy_b, classes, families)
        gb_cal = canon.fit_one(canon.GB_PAIR, train_pool, legacy_X, legacy_y, legacy_b, classes, families)

        rows = [
            s for s in test_rows
            if models.get(s["route"]) is not None and s["label"] in models[s["route"]][1].classes_
        ]

        fused_by_route = {route: fused_params(mt) for route, mt in models.items()}
        fold_stats = {name: Counter() for name in candidate_names}

        # Correctness gate: baseline canonical cached output is the reference.
        for s in rows:
            mt = models[s["route"]]
            x0 = features_baseline(s["data"])
            raw0 = score_standard(mt, x0)
            ctx0 = build_ctx_from_raw(mt, s, raw0)
            ref = downstream(cal, triad_cal, sig_cal, gb_cal, ctx0, classes, families)

            for fname, ffn in FEATURES.items():
                x = x0 if fname == "baseline" else ffn(s["data"])

                for scorer in ("standard", "fused"):
                    name = f"{fname}+{scorer}"
                    raw = (
                        score_standard(mt, x)
                        if scorer == "standard"
                        else score_fused(mt, x, fused_by_route[s["route"]])
                    )
                    ctx = build_ctx_from_raw(mt, s, raw)
                    out = downstream(cal, triad_cal, sig_cal, gb_cal, ctx, classes, families)

                    fs = fold_stats[name]
                    fs["n"] += 1
                    fs["hits"] += int(out and out[0] == s["label"])
                    if list(out) != list(ref):
                        fs["mismatch"] += 1
                    if not np.array_equal(x, x0):
                        fs["feature_array_diff"] += 1
                    if tuple(ctx.rank) != tuple(ctx0.rank):
                        fs["base_rank_diff"] += 1

        # Timing gate: time kernel+context+downstream only. Every candidate runs
        # on identical fitted objects and identical rows in this process.
        for name in candidate_names:
            fname, scorer = name.split("+")
            if fold_stats[name]["mismatch"] != 0:
                continue

            ffn = FEATURES[fname]
            elapsed = 0
            sink = 0
            for _ in range(TIMING_REPEATS):
                t0 = time.perf_counter_ns()
                for s in rows:
                    mt = models[s["route"]]
                    x = ffn(s["data"])
                    raw = (
                        score_standard(mt, x)
                        if scorer == "standard"
                        else score_fused(mt, x, fused_by_route[s["route"]])
                    )
                    ctx = build_ctx_from_raw(mt, s, raw)
                    out = downstream(cal, triad_cal, sig_cal, gb_cal, ctx, classes, families)
                    sink += len(out)
                elapsed += time.perf_counter_ns() - t0

            fs = fold_stats[name]
            fs["timed_calls"] = len(rows) * TIMING_REPEATS
            fs["elapsed_ns"] = elapsed
            fs["sink"] = sink

        fold_out = {"fold": fold, "n": len(rows), "candidates": {}}
        for name in candidate_names:
            fs = fold_stats[name]
            row = {
                "n": fs["n"],
                "hits": fs["hits"],
                "top1": fs["hits"] / max(1, fs["n"]),
                "mismatch_n": fs["mismatch"],
                "feature_array_diff_n": fs["feature_array_diff"],
                "base_rank_diff_n": fs["base_rank_diff"],
                "timed_calls": fs["timed_calls"],
                "ns_per_call": fs["elapsed_ns"] / max(1, fs["timed_calls"]) if fs["timed_calls"] else None,
            }
            fold_out["candidates"][name] = row
            for k, v in fs.items():
                agg[name][k] += v
        fold_rows.append(fold_out)

    pooled = {}
    for name in candidate_names:
        a = agg[name]
        pooled[name] = {
            "n": a["n"],
            "hits": a["hits"],
            "top1": a["hits"] / max(1, a["n"]),
            "mismatch_n": a["mismatch"],
            "feature_array_diff_n": a["feature_array_diff"],
            "base_rank_diff_n": a["base_rank_diff"],
            "timed_calls": a["timed_calls"],
            "ns_per_call": a["elapsed_ns"] / max(1, a["timed_calls"]) if a["timed_calls"] else None,
        }

    baseline_ns = pooled["baseline+standard"]["ns_per_call"]
    for name, row in pooled.items():
        if row["ns_per_call"] is not None:
            row["speedup_vs_baseline"] = baseline_ns / row["ns_per_call"]
            row["runtime_reduction_vs_baseline"] = 1.0 - row["ns_per_call"] / baseline_ns

    eligible = [
        (row["ns_per_call"], name)
        for name, row in pooled.items()
        if row["mismatch_n"] == 0 and row["hits"] == 365 and row["ns_per_call"] is not None
    ]
    winner = min(eligible)[1] if eligible else None

    print(json.dumps({
        "phase": "p2_multihop_kernel_tournament",
        "canonical_target": {"n": 418, "hits": 365, "top1": 365 / 418},
        "timing_repeats": TIMING_REPEATS,
        "pooled": pooled,
        "winner": winner,
        "folds": fold_rows,
        "collection_stats": dict(stats),
        "decision_rule": "Eligible only if final ranking mismatch_n=0 and hits=365/418; among eligible candidates choose lowest pooled ns_per_call.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
