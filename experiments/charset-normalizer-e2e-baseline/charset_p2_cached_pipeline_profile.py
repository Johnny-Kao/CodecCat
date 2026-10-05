from __future__ import annotations

import hashlib
import json
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
REPEATS = 9


def timed_ns(fn):
    t0 = time.perf_counter_ns()
    out = fn()
    return time.perf_counter_ns() - t0, out


def main():
    external, _, stats = base.collect_external()
    legacy_X, legacy_y, legacy_b = base.load_legacy_training()
    classes, families = calmod.build_vocab(legacy_y)
    paths = sorted(set(s["warc_path"] for s in external))
    fold_assign = {
        p: int.from_bytes(hashlib.sha256(("learning-curve:" + p).encode()).digest()[:8], "big") % OUTER_FOLDS
        for p in paths
    }

    total = Counter()
    folds = []

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

        feature_ns = linear_ns = byte_ns = downstream_ns = full_ns = 0
        sink = 0

        for _ in range(REPEATS):
            for s in rows:
                model_tuple = models[s["route"]]
                scaler, model = model_tuple

                dt, x = timed_ns(lambda: base.scorer_features(s["data"]))
                feature_ns += dt

                def do_linear():
                    scores = model.decision_function(scaler.transform(x[None, :]))
                    if scores.ndim == 1:
                        scores = np.column_stack([-scores, scores])
                    raw = scores[0]
                    order = np.argsort(raw)[::-1]
                    return raw, order
                dt, (raw, order) = timed_ns(do_linear)
                linear_ns += dt

                def do_byte():
                    data = s["data"]
                    sample = data[:4096]
                    arr = np.frombuffer(sample, dtype=np.uint8)
                    strict = rr.strict_utf8(data)
                    bom = rr.has_utf8_bom(data)
                    ascii_flag = rr.ascii_only(data)
                    high = float(np.mean(arr >= 128)) if len(arr) else 0.0
                    nul = float(np.mean(arr == 0)) if len(arr) else 0.0
                    rate = canon.replacement_rate(data)
                    return strict, bom, ascii_flag, high, nul, len(arr), rate
                dt, bytevals = timed_ns(do_byte)
                byte_ns += dt

                ctx = p1.InferenceContext(
                    data=s["data"],
                    route=s["route"],
                    rank=tuple(model.classes_[order]),
                    sorted_scores=raw[order],
                    score_map={c: float(v) for c, v in zip(model.classes_, raw)},
                    has_utf8_bom=bytevals[1],
                    strict_utf8=bytevals[0],
                    ascii_only=bytevals[2],
                    high_byte_ratio=bytevals[3],
                    nul_ratio=bytevals[4],
                    sample_len_4k=bytevals[5],
                    replacement_rate=bytevals[6],
                )

                def do_downstream():
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
                dt, out = timed_ns(do_downstream)
                downstream_ns += dt
                sink += len(out)

                dt, out2 = timed_ns(lambda: p1.cached_pipeline(
                    model_tuple, cal, triad_cal, sig_cal, gb_cal, s, classes, families
                ))
                full_ns += dt
                sink += len(out2)

        n_calls = len(rows) * REPEATS
        fold_row = {
            "fold": fold,
            "n": len(rows),
            "n_calls": n_calls,
            "feature_ns_per_call": feature_ns / max(1, n_calls),
            "linear_ns_per_call": linear_ns / max(1, n_calls),
            "byte_analysis_ns_per_call": byte_ns / max(1, n_calls),
            "downstream_ns_per_call": downstream_ns / max(1, n_calls),
            "full_cached_ns_per_call": full_ns / max(1, n_calls),
        }
        fold_row["accounted_ns_per_call"] = (
            fold_row["feature_ns_per_call"]
            + fold_row["linear_ns_per_call"]
            + fold_row["byte_analysis_ns_per_call"]
            + fold_row["downstream_ns_per_call"]
        )
        folds.append(fold_row)

        total["calls"] += n_calls
        total["feature_ns"] += feature_ns
        total["linear_ns"] += linear_ns
        total["byte_ns"] += byte_ns
        total["downstream_ns"] += downstream_ns
        total["full_ns"] += full_ns
        total["sink"] += sink

    calls = max(1, total["calls"])
    parts = {
        "scorer_features": total["feature_ns"] / calls,
        "linear_score_rank": total["linear_ns"] / calls,
        "byte_analysis": total["byte_ns"] / calls,
        "downstream_rerank_specialists": total["downstream_ns"] / calls,
    }
    accounted = sum(parts.values())
    full = total["full_ns"] / calls
    shares = {k: v / max(1.0, accounted) for k, v in parts.items()}

    print(json.dumps({
        "phase": "p2_cached_pipeline_profile",
        "repeats": REPEATS,
        "pooled": {
            "n_calls": total["calls"],
            "full_cached_ns_per_call": full,
            "accounted_ns_per_call": accounted,
            "parts_ns_per_call": parts,
            "parts_share_of_accounted": shares,
        },
        "folds": folds,
        "collection_stats": dict(stats),
        "next_step_rule": "Optimize the largest stable component first; preserve exact feature/ranking output and canonical 365/418 accuracy.",
        "sink": total["sink"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
