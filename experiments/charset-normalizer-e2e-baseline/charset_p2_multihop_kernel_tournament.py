from __future__ import annotations

import hashlib
import json
import math
import time

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


def features_combined(data: bytes) -> np.ndarray:
    data = hmt768(data)
    a = np.frombuffer(data, dtype=np.uint8)
    n = max(1, len(a))
    counts = np.bincount(a, minlength=256)
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
    scalars = np.array([
        math.log2(n + 1) / 16.0, high, nul, printable, lf, cr
    ], dtype=np.float32)
    return np.concatenate([unigram, bigram, scalars])


def fuse_scaler_linear(model_tuple):
    scaler, model = model_tuple
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    w = coef / scale[None, :]
    b = intercept - (coef * (mean / scale)[None, :]).sum(axis=1)
    return model.classes_, w, b


def fused_raw(fused, x):
    classes, w, b = fused
    z = np.asarray(x, dtype=np.float64) @ w.T + b
    if len(classes) == 2 and z.shape == (1,):
        s = float(z[0])
        return np.asarray([-s, s], dtype=np.float64)
    return np.asarray(z, dtype=np.float64)


def standard_raw(model_tuple, x):
    scaler, model = model_tuple
    scores = model.decision_function(scaler.transform(x[None, :]))
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores[0]


def byte_values(data, fast=False):
    sample = data[:4096]
    arr = np.frombuffer(sample, dtype=np.uint8)
    strict = rr.strict_utf8(data)
    bom = rr.has_utf8_bom(data)
    if fast:
        if len(arr):
            high_count = int(np.count_nonzero(arr >= 128))
            nul_count = int(np.count_nonzero(arr == 0))
            high = high_count / len(arr)
            nul = nul_count / len(arr)
            ascii_flag = high_count == 0
        else:
            high = nul = 0.0
            ascii_flag = True
    else:
        ascii_flag = rr.ascii_only(data)
        high = float(np.mean(arr >= 128)) if len(arr) else 0.0
        nul = float(np.mean(arr == 0)) if len(arr) else 0.0
    rate = canon.replacement_rate(data)
    return strict, bom, ascii_flag, high, nul, len(arr), rate


def build_ctx(model_tuple, s, feature_mode, linear_mode, fused_model, byte_fast):
    x = base.scorer_features(s["data"]) if feature_mode == "baseline" else features_combined(s["data"])
    raw = standard_raw(model_tuple, x) if linear_mode == "standard" else fused_raw(fused_model, x)
    _, model = model_tuple
    order = np.argsort(raw)[::-1]
    vals = byte_values(s["data"], fast=byte_fast)
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


def fuse_estimator(est):
    if est is None:
        return None
    scaler, model, *meta = est
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    w = coef / scale[None, :]
    b = intercept - (coef * (mean / scale)[None, :]).sum(axis=1)
    return model.classes_, w, b


class FastDownstream:
    def __init__(self, cal, triad_cal, sig_cal, gb_cal, classes, families):
        self.cal = fuse_estimator(cal)
        self.triad = fuse_estimator(triad_cal)
        self.sig = fuse_estimator(sig_cal)
        self.gb = fuse_estimator(gb_cal)
        self.class_idx = {c: i for i, c in enumerate(classes)}
        self.fam_idx = {c: i for i, c in enumerate(families)}
        self.classes = classes
        self.families = families

    def candidate_feature(self, ctx, candidate, rank_idx, score, top_score, second_score):
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
        cand_oh = [0.0] * len(self.classes)
        ci = self.class_idx.get(candidate)
        if ci is not None:
            cand_oh[ci] = 1.0
        fam_oh = [0.0] * len(self.families)
        fi = self.fam_idx.get(calmod.fam(candidate))
        if fi is not None:
            fam_oh[fi] = 1.0
        return np.asarray(vec + route_oh + cand_oh + fam_oh, dtype=np.float32)

    def choose_cal(self, ctx):
        if self.cal is None or len(ctx.rank) < 2:
            return list(ctx.rank)
        _, w, b = self.cal
        top_score = float(ctx.sorted_scores[0])
        second_score = float(ctx.sorted_scores[1])
        cand = list(ctx.rank[:calmod.TOP_CANDIDATES])
        X = np.stack([
            self.candidate_feature(
                ctx, c, i, float(ctx.sorted_scores[i]), top_score, second_score
            )
            for i, c in enumerate(cand)
        ])
        # Candidate calibrator is binary. P(class=1) is monotonic in its logit,
        # so argmax probability == argmax positive-class logit.
        z = X.astype(np.float64) @ w[0] + b[0]
        winner = int(np.argmax(z))
        out = list(ctx.rank)
        if winner != 0:
            chosen = out.pop(winner)
            out.insert(0, chosen)
        return out

    def hybrid(self, ctx):
        rank = list(ctx.rank)
        rule_rank = p1.cached_rule_rerank(ctx)
        cal_rank = self.choose_cal(ctx)
        return rule_rank if (rule_rank and rank and rule_rank[0] != rank[0]) else cal_rank

    def gate(self, ctx, hybrid):
        if self.triad is None or not hybrid or hybrid[0] not in tri.TRIAD:
            return hybrid
        if sum(c in tri.TRIAD for c in ctx.rank[:3]) < 2:
            return hybrid
        classes, w, b = self.triad
        x = p1.cached_triad_feature(ctx).astype(np.float64)
        logits = x @ w.T + b
        idx = int(np.argmax(logits))
        shifted = logits - np.max(logits)
        ex = np.exp(shifted)
        pmax = float(ex[idx] / ex.sum())
        pred = classes[idx]
        if pmax < 0.50 or pred not in ctx.rank[:3]:
            return hybrid
        out = list(hybrid)
        if pred in out:
            out.remove(pred)
            out.insert(0, pred)
        return out

    def pair(self, fused, pair, ctx, baseline):
        if fused is None or not baseline or baseline[0] not in pair:
            return baseline
        if not all(p in ctx.rank[:3] for p in pair):
            return baseline
        classes, w, b = fused
        x = p1.cached_pair_feature(ctx, pair).astype(np.float64)
        z = float(x @ w[0] + b[0])
        pred = classes[1] if z > 0.0 else classes[0]
        if pred not in ctx.rank[:3]:
            return baseline
        out = list(baseline)
        if pred in out:
            out.remove(pred)
            out.insert(0, pred)
        return out

    def run(self, ctx):
        hybrid = self.hybrid(ctx)
        baseline = self.gate(ctx, hybrid)
        sig = self.pair(self.sig, p1.SIG_PAIR, ctx, baseline)
        out = self.pair(self.gb, p1.GB_PAIR, ctx, sig)
        if (
            out and sig and out[0] != sig[0]
            and sig[0] == "gb18030" and out[0] == "utf-8"
            and ctx.replacement_rate > p1.RATE_THRESHOLD
        ):
            out = sig
        return out


def baseline_downstream(cal, triad_cal, sig_cal, gb_cal, ctx, classes, families):
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


CANDIDATES = {
    "baseline": dict(feature="baseline", linear="standard", byte_fast=False, downstream_fast=False),
    "downstream_fused": dict(feature="baseline", linear="standard", byte_fast=False, downstream_fast=True),
    "linear_fused": dict(feature="baseline", linear="fused", byte_fast=False, downstream_fast=False),
    "byte_fast": dict(feature="baseline", linear="standard", byte_fast=True, downstream_fast=False),
    "feature_combined": dict(feature="combined", linear="standard", byte_fast=False, downstream_fast=False),
    "downstream_linear": dict(feature="baseline", linear="fused", byte_fast=False, downstream_fast=True),
    "downstream_linear_byte": dict(feature="baseline", linear="fused", byte_fast=True, downstream_fast=True),
    "all_combined": dict(feature="combined", linear="fused", byte_fast=True, downstream_fast=True),
}


def main():
    external, _, stats = base.collect_external()
    legacy_X, legacy_y, legacy_b = base.load_legacy_training()
    classes, families = calmod.build_vocab(legacy_y)
    paths = sorted(set(s["warc_path"] for s in external))
    fold_assign = {
        p: int.from_bytes(hashlib.sha256(("learning-curve:" + p).encode()).digest()[:8], "big") % OUTER_FOLDS
        for p in paths
    }

    pooled = {
        name: dict(n=0, hits=0, mismatch=0, base_rank_diff=0, elapsed_ns=0, timed_calls=0)
        for name in CANDIDATES
    }
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
        fused_models = {route: fuse_scaler_linear(mt) for route, mt in models.items()}
        fast_ds = FastDownstream(cal, triad_cal, sig_cal, gb_cal, classes, families)

        fold_stats = {
            name: dict(n=0, hits=0, mismatch=0, base_rank_diff=0, elapsed_ns=0, timed_calls=0)
            for name in CANDIDATES
        }

        refs = []
        ref_ranks = []
        for s in rows:
            mt = models[s["route"]]
            ref_ctx = build_ctx(mt, s, "baseline", "standard", fused_models[s["route"]], False)
            ref = baseline_downstream(cal, triad_cal, sig_cal, gb_cal, ref_ctx, classes, families)
            refs.append(ref)
            ref_ranks.append(ref_ctx.rank)

        # Correctness gate.
        for s, ref, ref_rank in zip(rows, refs, ref_ranks):
            mt = models[s["route"]]
            for name, cfg in CANDIDATES.items():
                ctx = build_ctx(
                    mt, s, cfg["feature"], cfg["linear"], fused_models[s["route"]], cfg["byte_fast"]
                )
                out = fast_ds.run(ctx) if cfg["downstream_fast"] else baseline_downstream(
                    cal, triad_cal, sig_cal, gb_cal, ctx, classes, families
                )
                st = fold_stats[name]
                st["n"] += 1
                st["hits"] += int(out and out[0] == s["label"])
                st["mismatch"] += int(list(out) != list(ref))
                st["base_rank_diff"] += int(tuple(ctx.rank) != tuple(ref_rank))

        # Only correctness-safe candidates get timed.
        for name, cfg in CANDIDATES.items():
            st = fold_stats[name]
            if st["mismatch"] != 0:
                continue
            sink = 0
            t0 = time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    mt = models[s["route"]]
                    ctx = build_ctx(
                        mt, s, cfg["feature"], cfg["linear"], fused_models[s["route"]], cfg["byte_fast"]
                    )
                    out = fast_ds.run(ctx) if cfg["downstream_fast"] else baseline_downstream(
                        cal, triad_cal, sig_cal, gb_cal, ctx, classes, families
                    )
                    sink += len(out)
            elapsed = time.perf_counter_ns() - t0
            st["elapsed_ns"] = elapsed
            st["timed_calls"] = len(rows) * TIMING_REPEATS
            st["sink"] = sink

        fold_row = {"fold": fold, "n": len(rows), "candidates": {}}
        for name, st in fold_stats.items():
            fold_row["candidates"][name] = {
                "top1": st["hits"] / max(1, st["n"]),
                "mismatch_n": st["mismatch"],
                "base_rank_diff_n": st["base_rank_diff"],
                "ns_per_call": (
                    st["elapsed_ns"] / st["timed_calls"] if st["timed_calls"] else None
                ),
            }
            for key in ("n", "hits", "mismatch", "base_rank_diff", "elapsed_ns", "timed_calls"):
                pooled[name][key] += st[key]
        folds.append(fold_row)

    out = {}
    for name, st in pooled.items():
        out[name] = {
            "n": st["n"],
            "hits": st["hits"],
            "top1": st["hits"] / max(1, st["n"]),
            "mismatch_n": st["mismatch"],
            "base_rank_diff_n": st["base_rank_diff"],
            "ns_per_call": st["elapsed_ns"] / st["timed_calls"] if st["timed_calls"] else None,
        }

    baseline_ns = out["baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_baseline"] = baseline_ns / row["ns_per_call"]
            row["runtime_reduction_vs_baseline"] = 1.0 - row["ns_per_call"] / baseline_ns

    eligible = [
        (row["ns_per_call"], name)
        for name, row in out.items()
        if row["mismatch_n"] == 0
        and row["hits"] == 365
        and row["ns_per_call"] is not None
    ]
    winner = min(eligible)[1] if eligible else None

    print(json.dumps({
        "phase": "p2_full_pipeline_multihop_tournament",
        "profile_basis": {
            "downstream_share": 0.5237,
            "linear_share": 0.2430,
            "byte_analysis_share": 0.1515,
            "scorer_features_share": 0.0818,
        },
        "canonical_target": {"n": 418, "hits": 365, "top1": 365 / 418},
        "timing_repeats": TIMING_REPEATS,
        "pooled": out,
        "winner": winner,
        "folds": folds,
        "collection_stats": dict(stats),
        "decision_rule": "Candidate must preserve exact final ranking on all 418 samples and 365 hits; choose fastest eligible pooled runtime.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
