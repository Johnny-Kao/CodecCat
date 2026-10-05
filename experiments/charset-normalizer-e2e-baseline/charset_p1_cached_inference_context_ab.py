from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

import charset_external_scorer_learning_curve as lc
import charset_external_scorer_domain_shift_ab as base
import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_post_gate_050_error_decomposition as gate050
import charset_pair_specialist_ab as pairmod
import charset_minimal_reranker_ab as rr
import charset_canonical_guarded_final_validation as canon

OUTER_FOLDS = 4
TRAIN_N = 300
SIG_PAIR = ("utf-8", "utf-8-sig")
GB_PAIR = ("utf-8", "gb18030")
RATE_THRESHOLD = 0.02
TIMING_REPEATS = 7


def fit_one(pair, *args):
    old = pairmod.PAIR
    pairmod.PAIR = pair
    try:
        return pairmod.fit_pair(*args)
    finally:
        pairmod.PAIR = old


def apply_one(pair_cal, pair, model, s, baseline):
    old = pairmod.PAIR
    pairmod.PAIR = pair
    try:
        return pairmod.apply_pair(pair_cal, model, s, baseline)
    finally:
        pairmod.PAIR = old


def replacement_rate(data):
    if not data:
        return 0.0
    txt = data.decode("utf-8", errors="replace")
    return txt.count("\ufffd") / max(1, len(txt))


def old_pipeline(model, cal, triad_cal, sig_cal, gb_cal, s, classes, families):
    _, hybrid = tri.make_hybrid(model, cal, s, classes, families)
    baseline = gate050.apply_gated(triad_cal, model, s, hybrid)
    sig = canon.apply_one(sig_cal, canon.SIG_PAIR, model, s, baseline)
    return canon.apply_guarded_gb(gb_cal, model, s, sig)


@dataclass(frozen=True)
class InferenceContext:
    data: bytes
    route: str
    rank: tuple[str, ...]
    sorted_scores: np.ndarray
    score_map: dict[str, float]
    has_utf8_bom: bool
    strict_utf8: bool
    ascii_only: bool
    high_byte_ratio: float
    nul_ratio: float
    sample_len_4k: int
    replacement_rate: float


def build_context(model_tuple, s):
    data = s["data"]
    scaler, model = model_tuple
    x = base.scorer_features(data)
    scores = model.decision_function(scaler.transform(x[None, :]))
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    raw = scores[0]
    order = np.argsort(raw)[::-1]
    rank = tuple(model.classes_[order])
    sorted_scores = raw[order]

    sample = data[:4096]
    arr = np.frombuffer(sample, dtype=np.uint8)
    strict = rr.strict_utf8(data)
    bom = rr.has_utf8_bom(data)
    ascii_flag = rr.ascii_only(data)
    high = float(np.mean(arr >= 128)) if len(arr) else 0.0
    nul = float(np.mean(arr == 0)) if len(arr) else 0.0

    return InferenceContext(
        data=data,
        route=s["route"],
        rank=rank,
        sorted_scores=sorted_scores,
        score_map={c: float(v) for c, v in zip(model.classes_, raw)},
        has_utf8_bom=bom,
        strict_utf8=strict,
        ascii_only=ascii_flag,
        high_byte_ratio=high,
        nul_ratio=nul,
        sample_len_4k=len(arr),
        replacement_rate=replacement_rate(data),
    )


def cached_rule_rerank(ctx):
    r = list(ctx.rank)
    if not r:
        return r

    if ctx.has_utf8_bom and "utf-8-sig" in r:
        r.remove("utf-8-sig")
        r.insert(0, "utf-8-sig")
        return r

    if ctx.strict_utf8 and "utf-8" in r:
        if r[0] != "utf-8":
            topfam = rr.family(r[0])
            if topfam in ("utf", "cjk", "singlebyte-win", "singlebyte-iso", "ascii"):
                r.remove("utf-8")
                r.insert(0, "utf-8")
        return r

    if ctx.ascii_only and "ascii" in r:
        r.remove("ascii")
        r.insert(0, "ascii")
        return r

    top3 = r[:3]
    if r[0] == "iso-8859-9" and "iso-8859-1" in top3:
        r.remove("iso-8859-1")
        r.insert(0, "iso-8859-1")
    elif r[0] == "cp1257" and "iso-8859-1" in top3:
        r.remove("iso-8859-1")
        r.insert(0, "iso-8859-1")
    return r


def cached_candidate_feature(ctx, candidate, rank_idx, score, top_score, second_score, classes, families):
    class_idx = {c: i for i, c in enumerate(classes)}
    fam_idx = {c: i for i, c in enumerate(families)}
    vec = [
        float(score),
        float(score - top_score),
        float(score - second_score),
        float(rank_idx),
        float(ctx.has_utf8_bom),
        float(ctx.strict_utf8),
        float(ctx.ascii_only),
    ]
    route_oh = [1.0 if ctx.route == r else 0.0 for r in calmod.ROUTES]
    cand_oh = [0.0] * len(classes)
    if candidate in class_idx:
        cand_oh[class_idx[candidate]] = 1.0
    fam = calmod.fam(candidate)
    fam_oh = [0.0] * len(families)
    if fam in fam_idx:
        fam_oh[fam_idx[fam]] = 1.0
    return np.asarray(vec + route_oh + cand_oh + fam_oh, dtype=np.float32)


def cached_calibrator(cal, ctx, classes, families):
    if len(ctx.rank) < 2:
        return list(ctx.rank)
    scaler, model, _, _ = cal
    top_score = float(ctx.sorted_scores[0])
    second_score = float(ctx.sorted_scores[1])
    cand = list(ctx.rank[:calmod.TOP_CANDIDATES])
    X = np.stack([
        cached_candidate_feature(
            ctx,
            c,
            i,
            float(ctx.sorted_scores[i]),
            top_score,
            second_score,
            classes,
            families,
        )
        for i, c in enumerate(cand)
    ])
    p = model.predict_proba(scaler.transform(X))[:, 1]
    winner = int(np.argmax(p))
    out = list(ctx.rank)
    if winner != 0:
        chosen = out.pop(winner)
        out.insert(0, chosen)
    return out


def cached_hybrid(cal, ctx, classes, families):
    rank = list(ctx.rank)
    rule_rank = cached_rule_rerank(ctx)
    cal_rank = cached_calibrator(cal, ctx, classes, families)
    return rule_rank if (rule_rank and rank and rule_rank[0] != rank[0]) else cal_rank


def cached_triad_feature(ctx):
    vals = [ctx.score_map.get(c, -20.0) for c in tri.TRIAD]
    margins = [
        vals[0] - vals[1],
        vals[0] - vals[2],
        vals[1] - vals[2],
        max(vals) - sorted(vals)[-2] if len(vals) >= 2 else 0.0,
    ]
    scalars = [
        float(ctx.strict_utf8),
        float(ctx.has_utf8_bom),
        float(ctx.ascii_only),
        ctx.high_byte_ratio,
        ctx.nul_ratio,
        float(ctx.sample_len_4k),
    ]
    route_oh = [1.0 if ctx.route == r else 0.0 for r in ("U", "N", "RL", "RH")]
    return np.asarray(vals + margins + scalars + route_oh, dtype=np.float32)


def cached_gate(triad_cal, ctx, hybrid):
    if triad_cal is None or not hybrid or hybrid[0] not in tri.TRIAD:
        return hybrid
    if sum(c in tri.TRIAD for c in ctx.rank[:3]) < 2:
        return hybrid
    sc, clf, _, _ = triad_cal
    x = cached_triad_feature(ctx)
    probs = clf.predict_proba(sc.transform(x[None, :]))[0]
    idx = int(probs.argmax())
    pred = clf.classes_[idx]
    p = float(probs[idx])
    if p < gate050.CONF or pred not in ctx.rank[:3]:
        return hybrid
    out = list(hybrid)
    if pred in out:
        out.remove(pred)
        out.insert(0, pred)
    return out


def cached_pair_feature(ctx, pair):
    a, b = pair
    sa, sb = ctx.score_map.get(a, -20.0), ctx.score_map.get(b, -20.0)
    scalars = [
        sa,
        sb,
        sa - sb,
        abs(sa - sb),
        float(ctx.strict_utf8),
        float(ctx.has_utf8_bom),
        float(ctx.ascii_only),
        ctx.high_byte_ratio,
        ctx.nul_ratio,
        float(ctx.sample_len_4k),
    ]
    route_oh = [1.0 if ctx.route == r else 0.0 for r in ("U", "N", "RL", "RH")]
    return np.asarray(scalars + route_oh, dtype=np.float32)


def cached_apply_pair(pair_cal, pair, ctx, baseline):
    if pair_cal is None or not baseline or baseline[0] not in pair:
        return baseline
    if not all(p in ctx.rank[:3] for p in pair):
        return baseline
    sc, clf, _, _ = pair_cal
    pred = clf.predict(sc.transform(cached_pair_feature(ctx, pair)[None, :]))[0]
    if pred not in ctx.rank[:3]:
        return baseline
    out = list(baseline)
    if pred in out:
        out.remove(pred)
        out.insert(0, pred)
    return out


def cached_pipeline(model, cal, triad_cal, sig_cal, gb_cal, s, classes, families):
    ctx = build_context(model, s)
    hybrid = cached_hybrid(cal, ctx, classes, families)
    baseline = cached_gate(triad_cal, ctx, hybrid)
    sig = cached_apply_pair(sig_cal, SIG_PAIR, ctx, baseline)
    out = cached_apply_pair(gb_cal, GB_PAIR, ctx, sig)
    if (
        out
        and sig
        and out[0] != sig[0]
        and sig[0] == "gb18030"
        and out[0] == "utf-8"
        and ctx.replacement_rate > RATE_THRESHOLD
    ):
        out = sig
    return out


def main():
    external, _, stats = base.collect_external()
    legacy_X, legacy_y, legacy_b = base.load_legacy_training()
    classes, families = calmod.build_vocab(legacy_y)
    paths = sorted(set(s["warc_path"] for s in external))
    fold_assign = {
        p: int.from_bytes(hashlib.sha256(("learning-curve:" + p).encode()).digest()[:8], "big") % OUTER_FOLDS
        for p in paths
    }

    original_scorer_features = base.scorer_features
    feature_calls = Counter()
    mode = {"name": "none"}

    def counted_scorer_features(data):
        feature_calls[mode["name"]] += 1
        return original_scorer_features(data)

    folds = []
    mismatches = []
    total = Counter()
    timing_rows = []

    try:
        for fold in range(OUTER_FOLDS):
            train_pool = [s for s in external if fold_assign[s["warc_path"]] != fold]
            test_rows = [s for s in external if fold_assign[s["warc_path"]] == fold]
            ext_train = lc.deterministic_nested_subset(train_pool, TRAIN_N)

            # Keep training byte-for-byte on the canonical path. Instrument only
            # inference, after all fitted objects are finalized.
            base.scorer_features = original_scorer_features
            models = lc.fit_route_models(ext_train, legacy_X, legacy_y, legacy_b)
            cal = calmod.crossfit_calibration_rows(train_pool, legacy_X, legacy_y, legacy_b, classes, families)
            triad_cal = tri.crossfit_triad_rows(train_pool, legacy_X, legacy_y, legacy_b, classes, families)
            sig_cal = canon.fit_one(canon.SIG_PAIR, train_pool, legacy_X, legacy_y, legacy_b, classes, families)
            gb_cal = canon.fit_one(canon.GB_PAIR, train_pool, legacy_X, legacy_y, legacy_b, classes, families)

            base.scorer_features = counted_scorer_features

            valid_rows = [
                s for s in test_rows
                if models.get(s["route"]) is not None and s["label"] in models[s["route"]][1].classes_
            ]

            old_outputs = []
            mode["name"] = "old"
            t0 = time.perf_counter_ns()
            for s in valid_rows:
                old_outputs.append(old_pipeline(
                    models[s["route"]], cal, triad_cal, sig_cal, gb_cal, s, classes, families
                ))
            old_ns = time.perf_counter_ns() - t0

            cached_outputs = []
            mode["name"] = "cached"
            t0 = time.perf_counter_ns()
            for s in valid_rows:
                cached_outputs.append(cached_pipeline(
                    models[s["route"]], cal, triad_cal, sig_cal, gb_cal, s, classes, families
                ))
            cached_ns = time.perf_counter_ns() - t0

            for i, (s, old, new) in enumerate(zip(valid_rows, old_outputs, cached_outputs)):
                total["n"] += 1
                total["hits"] += int(new and new[0] == s["label"])
                if old != new:
                    mismatches.append({
                        "fold": fold,
                        "index": i,
                        "host": s.get("host", ""),
                        "truth": s["label"],
                        "old": old[:5] if old else [],
                        "cached": new[:5] if new else [],
                    })

            folds.append({
                "fold": fold,
                "n": len(valid_rows),
                "old_ns_per_sample": old_ns / max(1, len(valid_rows)),
                "cached_ns_per_sample": cached_ns / max(1, len(valid_rows)),
            })

            # Timing repeats after equivalence pass. Keep outputs live to avoid
            # measuring only Python loop overhead.
            old_acc = cached_acc = 0
            mode["name"] = "old_timing"
            t0 = time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for s in valid_rows:
                    out = old_pipeline(models[s["route"]], cal, triad_cal, sig_cal, gb_cal, s, classes, families)
                    old_acc += len(out)
            old_repeat_ns = time.perf_counter_ns() - t0

            mode["name"] = "cached_timing"
            t0 = time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for s in valid_rows:
                    out = cached_pipeline(models[s["route"]], cal, triad_cal, sig_cal, gb_cal, s, classes, families)
                    cached_acc += len(out)
            cached_repeat_ns = time.perf_counter_ns() - t0
            assert old_acc == cached_acc

            timing_rows.append({
                "fold": fold,
                "n_calls": len(valid_rows) * TIMING_REPEATS,
                "old_ns_per_call": old_repeat_ns / max(1, len(valid_rows) * TIMING_REPEATS),
                "cached_ns_per_call": cached_repeat_ns / max(1, len(valid_rows) * TIMING_REPEATS),
                "speedup": old_repeat_ns / max(1, cached_repeat_ns),
            })
    finally:
        base.scorer_features = original_scorer_features

    old_calls = feature_calls["old"]
    cached_calls = feature_calls["cached"]
    pooled_old_timing = sum(x["old_ns_per_call"] * x["n_calls"] for x in timing_rows)
    pooled_cached_timing = sum(x["cached_ns_per_call"] * x["n_calls"] for x in timing_rows)
    pooled_calls = sum(x["n_calls"] for x in timing_rows)

    print(json.dumps({
        "phase": "p1_cached_inference_context_ab",
        "equivalence": {
            "n": total["n"],
            "mismatch_n": len(mismatches),
            "mismatches": mismatches[:20],
            "cached_hits": total["hits"],
            "cached_top1": total["hits"] / max(1, total["n"]),
        },
        "scorer_feature_calls_equivalence_pass": {
            "old": old_calls,
            "cached": cached_calls,
            "old_per_sample": old_calls / max(1, total["n"]),
            "cached_per_sample": cached_calls / max(1, total["n"]),
            "reduction_ratio": 1.0 - cached_calls / max(1, old_calls),
        },
        "timing": {
            "repeats": TIMING_REPEATS,
            "folds": timing_rows,
            "pooled_old_ns_per_call": pooled_old_timing / max(1, pooled_calls),
            "pooled_cached_ns_per_call": pooled_cached_timing / max(1, pooled_calls),
            "pooled_speedup": pooled_old_timing / max(1, pooled_cached_timing),
        },
        "folds": folds,
        "collection_stats": dict(stats),
        "acceptance": {
            "exact_output_equivalence_required": True,
            "target_cached_scorer_feature_calls_per_sample": 1.0,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
