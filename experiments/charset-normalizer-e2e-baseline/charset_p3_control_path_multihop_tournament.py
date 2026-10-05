from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np

import charset_external_scorer_learning_curve as lc
import charset_external_scorer_domain_shift_ab as base
import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_canonical_guarded_final_validation as canon
import charset_minimal_reranker_ab as rr
import charset_p1_cached_inference_context_ab as p1
import charset_p2_multihop_kernel_tournament as p2

OUTER_FOLDS = 4
TRAIN_N = 300
TIMING_REPEATS = 9


@dataclass(frozen=True)
class LiteContext:
    data: bytes
    route: str
    rank: tuple[str, ...]
    sorted_scores: np.ndarray
    raw_scores: np.ndarray
    classes: np.ndarray
    has_utf8_bom: bool
    strict_utf8: bool
    ascii_only: bool
    high_byte_ratio: float
    nul_ratio: float
    sample_len_4k: int
    replacement_rate: float


def byte_values_single_decode(data):
    sample = data[:4096]
    arr = np.frombuffer(sample, dtype=np.uint8)
    if len(arr):
        high_count = int(np.count_nonzero(arr >= 128))
        nul_count = int(np.count_nonzero(arr == 0))
        high = high_count / len(arr)
        nul = nul_count / len(arr)
        ascii_flag = high_count == 0
    else:
        high = nul = 0.0
        ascii_flag = True

    bom = data.startswith(b"\xef\xbb\xbf")
    try:
        data.decode("utf-8", errors="strict")
        strict = True
        rate = 0.0
    except UnicodeDecodeError:
        strict = False
        txt = data.decode("utf-8", errors="replace")
        rate = txt.count("\ufffd") / max(1, len(txt))
    return strict, bom, ascii_flag, high, nul, len(arr), rate


def build_lite(model_tuple, fused_model, s, single_decode):
    x = p2.features_combined(s["data"])
    raw = p2.fused_raw(fused_model, x)
    _, model = model_tuple
    order = np.argsort(raw)[::-1]
    vals = (
        byte_values_single_decode(s["data"])
        if single_decode
        else p2.byte_values(s["data"], fast=True)
    )
    return LiteContext(
        data=s["data"],
        route=s["route"],
        rank=tuple(model.classes_[order]),
        sorted_scores=np.asarray(raw)[order],
        raw_scores=np.asarray(raw),
        classes=model.classes_,
        has_utf8_bom=vals[1],
        strict_utf8=vals[0],
        ascii_only=vals[2],
        high_byte_ratio=vals[3],
        nul_ratio=vals[4],
        sample_len_4k=vals[5],
        replacement_rate=vals[6],
    )


class FastDownstreamV2:
    def __init__(self, cal, triad_cal, sig_cal, gb_cal, classes, families, route_models, precompute_layout):
        self.cal = p2.fuse_estimator(cal)
        self.triad = p2.fuse_estimator(triad_cal)
        self.sig = p2.fuse_estimator(sig_cal)
        self.gb = p2.fuse_estimator(gb_cal)
        self.classes = classes
        self.families = families
        self.class_idx = {c: i for i, c in enumerate(classes)}
        self.fam_idx = {c: i for i, c in enumerate(families)}
        self.precompute_layout = precompute_layout

        self.route_pos = {r: i for i, r in enumerate(calmod.ROUTES)}
        self.model_class_idx = {
            route: {c: i for i, c in enumerate(mt[1].classes_)}
            for route, mt in route_models.items()
        }

        self.candidate_suffix = {}
        if precompute_layout:
            suffix_len = len(calmod.ROUTES) + len(classes) + len(families)
            for route in calmod.ROUTES:
                for cand in classes:
                    suffix = np.zeros(suffix_len, dtype=np.float32)
                    rp = self.route_pos.get(route)
                    if rp is not None:
                        suffix[rp] = 1.0
                    ci = self.class_idx.get(cand)
                    if ci is not None:
                        suffix[len(calmod.ROUTES) + ci] = 1.0
                    fi = self.fam_idx.get(calmod.fam(cand))
                    if fi is not None:
                        suffix[len(calmod.ROUTES) + len(classes) + fi] = 1.0
                    self.candidate_suffix[(route, cand)] = suffix

        self.triad_model_idx = {
            route: [self.model_class_idx[route].get(c, -1) for c in tri.TRIAD]
            for route in self.model_class_idx
        }
        self.pair_model_idx = {}
        for route in self.model_class_idx:
            self.pair_model_idx[(route, p1.SIG_PAIR)] = [
                self.model_class_idx[route].get(c, -1) for c in p1.SIG_PAIR
            ]
            self.pair_model_idx[(route, p1.GB_PAIR)] = [
                self.model_class_idx[route].get(c, -1) for c in p1.GB_PAIR
            ]

    def rule_rerank(self, ctx):
        r = list(ctx.rank)
        if not r:
            return r
        if ctx.has_utf8_bom and "utf-8-sig" in r:
            r.remove("utf-8-sig"); r.insert(0, "utf-8-sig"); return r
        if ctx.strict_utf8 and "utf-8" in r:
            if r[0] != "utf-8":
                topfam = rr.family(r[0])
                if topfam in ("utf", "cjk", "singlebyte-win", "singlebyte-iso", "ascii"):
                    r.remove("utf-8"); r.insert(0, "utf-8")
            return r
        if ctx.ascii_only and "ascii" in r:
            r.remove("ascii"); r.insert(0, "ascii"); return r
        top3 = r[:3]
        if r[0] == "iso-8859-9" and "iso-8859-1" in top3:
            r.remove("iso-8859-1"); r.insert(0, "iso-8859-1")
        elif r[0] == "cp1257" and "iso-8859-1" in top3:
            r.remove("iso-8859-1"); r.insert(0, "iso-8859-1")
        return r

    def candidate_feature(self, ctx, candidate, rank_idx, score, top_score, second_score):
        head = np.asarray([
            float(score),
            float(score - top_score),
            float(score - second_score),
            float(rank_idx),
            float(ctx.has_utf8_bom),
            float(ctx.strict_utf8),
            float(ctx.ascii_only),
        ], dtype=np.float32)

        if self.precompute_layout:
            return np.concatenate([head, self.candidate_suffix[(ctx.route, candidate)]])

        route_oh = [1.0 if ctx.route == r else 0.0 for r in calmod.ROUTES]
        cand_oh = [0.0] * len(self.classes)
        ci = self.class_idx.get(candidate)
        if ci is not None:
            cand_oh[ci] = 1.0
        fam_oh = [0.0] * len(self.families)
        fi = self.fam_idx.get(calmod.fam(candidate))
        if fi is not None:
            fam_oh[fi] = 1.0
        return np.asarray(head.tolist() + route_oh + cand_oh + fam_oh, dtype=np.float32)

    def choose_cal(self, ctx):
        if self.cal is None or len(ctx.rank) < 2:
            return list(ctx.rank)
        _, w, b = self.cal
        top = float(ctx.sorted_scores[0])
        second = float(ctx.sorted_scores[1])
        cand = list(ctx.rank[:calmod.TOP_CANDIDATES])
        X = np.stack([
            self.candidate_feature(ctx, c, i, float(ctx.sorted_scores[i]), top, second)
            for i, c in enumerate(cand)
        ])
        z = X.astype(np.float64) @ w[0] + b[0]
        winner = int(np.argmax(z))
        out = list(ctx.rank)
        if winner != 0:
            chosen = out.pop(winner); out.insert(0, chosen)
        return out

    def hybrid(self, ctx):
        rank = list(ctx.rank)
        rule = self.rule_rerank(ctx)
        cal = self.choose_cal(ctx)
        return rule if (rule and rank and rule[0] != rank[0]) else cal

    def triad_feature(self, ctx):
        idxs = self.triad_model_idx[ctx.route]
        vals = [float(ctx.raw_scores[i]) if i >= 0 else -20.0 for i in idxs]
        margins = [
            vals[0] - vals[1],
            vals[0] - vals[2],
            vals[1] - vals[2],
            max(vals) - sorted(vals)[-2],
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

    def gate(self, ctx, hybrid):
        if self.triad is None or not hybrid or hybrid[0] not in tri.TRIAD:
            return hybrid
        if sum(c in tri.TRIAD for c in ctx.rank[:3]) < 2:
            return hybrid
        classes, w, b = self.triad
        x = self.triad_feature(ctx).astype(np.float64)
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
            out.remove(pred); out.insert(0, pred)
        return out

    def pair_feature(self, ctx, pair):
        idxs = self.pair_model_idx[(ctx.route, pair)]
        sa = float(ctx.raw_scores[idxs[0]]) if idxs[0] >= 0 else -20.0
        sb = float(ctx.raw_scores[idxs[1]]) if idxs[1] >= 0 else -20.0
        scalars = [
            sa, sb, sa - sb, abs(sa - sb),
            float(ctx.strict_utf8),
            float(ctx.has_utf8_bom),
            float(ctx.ascii_only),
            ctx.high_byte_ratio,
            ctx.nul_ratio,
            float(ctx.sample_len_4k),
        ]
        route_oh = [1.0 if ctx.route == r else 0.0 for r in ("U", "N", "RL", "RH")]
        return np.asarray(scalars + route_oh, dtype=np.float32)

    def pair(self, fused, pair, ctx, baseline):
        if fused is None or not baseline or baseline[0] not in pair:
            return baseline
        if not all(p in ctx.rank[:3] for p in pair):
            return baseline
        classes, w, b = fused
        x = self.pair_feature(ctx, pair).astype(np.float64)
        z = float(x @ w[0] + b[0])
        pred = classes[1] if z > 0.0 else classes[0]
        if pred not in ctx.rank[:3]:
            return baseline
        out = list(baseline)
        if pred in out:
            out.remove(pred); out.insert(0, pred)
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


CANDIDATES = {
    "p2_baseline": dict(single_decode=False, raw_index=False, precompute=False),
    "single_decode": dict(single_decode=True, raw_index=False, precompute=False),
    "raw_index": dict(single_decode=False, raw_index=True, precompute=False),
    "precompute_layout": dict(single_decode=False, raw_index=False, precompute=True),
    "raw_index_precompute": dict(single_decode=False, raw_index=True, precompute=True),
    "all_p3": dict(single_decode=True, raw_index=True, precompute=True),
}


def run_p2_reference(model_tuple, fused_model, fast_ds, s):
    ctx = p2.build_ctx(model_tuple, s, "combined", "fused", fused_model, True)
    return fast_ds.run(ctx)


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
        name: dict(n=0, hits=0, mismatch=0, elapsed_ns=0, timed_calls=0)
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
        fused_models = {route: p2.fuse_scaler_linear(mt) for route, mt in models.items()}
        p2_ds = p2.FastDownstream(cal, triad_cal, sig_cal, gb_cal, classes, families)

        refs = [
            run_p2_reference(models[s["route"]], fused_models[s["route"]], p2_ds, s)
            for s in rows
        ]

        fold_stats = {}
        for name, cfg in CANDIDATES.items():
            ds = FastDownstreamV2(
                cal, triad_cal, sig_cal, gb_cal, classes, families, models, cfg["precompute"]
            )
            mismatch = hits = 0
            outputs = []
            for s, ref in zip(rows, refs):
                if cfg["raw_index"]:
                    ctx = build_lite(
                        models[s["route"]], fused_models[s["route"]], s, cfg["single_decode"]
                    )
                    out = ds.run(ctx)
                else:
                    ctx = p2.build_ctx(
                        models[s["route"]], s, "combined", "fused",
                        fused_models[s["route"]], cfg["single_decode"]
                    )
                    if cfg["precompute"]:
                        # Use LiteContext only for layout optimization so raw score
                        # indexing remains isolated to the raw_index candidates.
                        lctx = build_lite(
                            models[s["route"]], fused_models[s["route"]], s, cfg["single_decode"]
                        )
                        out = ds.run(lctx)
                    else:
                        out = p2_ds.run(ctx)
                outputs.append(out)
                mismatch += int(list(out) != list(ref))
                hits += int(out and out[0] == s["label"])

            elapsed_ns = timed_calls = 0
            if mismatch == 0:
                t0 = time.perf_counter_ns()
                sink = 0
                for _ in range(TIMING_REPEATS):
                    for s in rows:
                        if cfg["raw_index"] or cfg["precompute"]:
                            ctx = build_lite(
                                models[s["route"]], fused_models[s["route"]], s, cfg["single_decode"]
                            )
                            out = ds.run(ctx)
                        else:
                            ctx = p2.build_ctx(
                                models[s["route"]], s, "combined", "fused",
                                fused_models[s["route"]], cfg["single_decode"]
                            )
                            out = p2_ds.run(ctx)
                        sink += len(out)
                elapsed_ns = time.perf_counter_ns() - t0
                timed_calls = len(rows) * TIMING_REPEATS

            fold_stats[name] = {
                "n": len(rows),
                "hits": hits,
                "mismatch_n": mismatch,
                "ns_per_call": elapsed_ns / timed_calls if timed_calls else None,
            }
            pooled[name]["n"] += len(rows)
            pooled[name]["hits"] += hits
            pooled[name]["mismatch"] += mismatch
            pooled[name]["elapsed_ns"] += elapsed_ns
            pooled[name]["timed_calls"] += timed_calls

        folds.append({"fold": fold, "n": len(rows), "candidates": fold_stats})

    out = {}
    for name, st in pooled.items():
        out[name] = {
            "n": st["n"],
            "hits": st["hits"],
            "top1": st["hits"] / max(1, st["n"]),
            "mismatch_n": st["mismatch"],
            "ns_per_call": (
                st["elapsed_ns"] / st["timed_calls"] if st["timed_calls"] else None
            ),
        }

    base_ns = out["p2_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p2"] = base_ns / row["ns_per_call"]
            row["runtime_reduction_vs_p2"] = 1.0 - row["ns_per_call"] / base_ns

    eligible = [
        (row["ns_per_call"], name)
        for name, row in out.items()
        if row["mismatch_n"] == 0 and row["hits"] == 365 and row["ns_per_call"] is not None
    ]
    winner = min(eligible)[1] if eligible else None

    print(json.dumps({
        "phase": "p3_control_path_multihop_tournament",
        "canonical_target": {"n": 418, "hits": 365, "top1": 365 / 418},
        "timing_repeats": TIMING_REPEATS,
        "pooled": out,
        "winner": winner,
        "folds": folds,
        "collection_stats": dict(stats),
        "decision_rule": "Preserve exact P2 final ranking on all 418 samples and 365 hits; choose fastest pooled candidate.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
