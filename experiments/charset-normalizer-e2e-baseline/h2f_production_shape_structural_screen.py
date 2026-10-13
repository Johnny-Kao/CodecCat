from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from codeccat import load_bundle
from codeccat.features import baseline_from_array, hmt768_array
from codeccat.runtime import _raw, _signals

from h1_structural_validity_probe import VALIDATORS
from h2_fused_candidate_mask_ab import collect, dedupe, normalize

sys.path.insert(0, str(Path(__file__).parent))
import h2f_structural_mask

BITS = {
    "big5": 1,
    "gb18030": 2,
    "shift-jis": 4,
    "euc-jp": 8,
    "euc-kr": 16,
}


def python_invalid_mask(data: bytes) -> int:
    out = 0
    for label, fn in VALIDATORS.items():
        if not fn(data):
            out |= BITS[label]
    return out


class ScreenCache:
    def __init__(self, bundle):
        self.cache = {}
        for route, model in bundle.baseline.route_models.items():
            route_cache = {}
            for mask in range(32):
                idx = np.asarray(
                    [
                        i for i, label in enumerate(model.classes)
                        if not (label in BITS and (mask & BITS[label]))
                    ],
                    dtype=np.intp,
                )
                route_cache[mask] = (
                    np.ascontiguousarray(model.weight[idx], dtype=np.float64),
                    np.ascontiguousarray(model.bias[idx], dtype=np.float64),
                    tuple(model.classes[i] for i in idx),
                )
            self.cache[route] = route_cache

    def score(self, route, mask, vector):
        w, b, classes = self.cache[route][mask]
        x = np.asarray(vector, dtype=np.float64)
        raw = np.dot(w, x) + b
        return np.asarray(raw, dtype=np.float64), classes


def rank_raw(model, vector):
    raw = _raw(model, vector)
    order = raw.argsort()[::-1]
    return tuple(np.asarray(model.classes, dtype=object)[order])


def rank_screened(cache, route, mask, vector):
    raw, classes = cache.score(route, mask, vector)
    order = raw.argsort()[::-1]
    return tuple(np.asarray(classes, dtype=object)[order])


def semantic_and_accuracy(rows, bundle, cache):
    mask_mismatch = 0
    truth_eliminated = 0
    changed = beneficial = harmful = neutral = 0
    base_hits = screen_hits = 0
    rows_before = rows_after = 0
    mask_counts = Counter()
    routes = defaultdict(Counter)

    for s in rows:
        pmask = python_invalid_mask(s["data"])
        cmask = h2f_structural_mask.invalid_mask(s["data"])
        if pmask != cmask:
            mask_mismatch += 1

        truth = normalize(s["label"])
        if truth in BITS and (cmask & BITS[truth]):
            truth_eliminated += 1

        signals = _signals(s["data"])
        vector = baseline_from_array(hmt768_array(s["data"]))
        model = bundle.baseline.route_models[signals.route]

        a = rank_raw(model, vector)
        b = rank_screened(cache, signals.route, cmask, vector)
        ah = int(bool(a) and a[0] == truth)
        bh = int(bool(b) and b[0] == truth)
        base_hits += ah
        screen_hits += bh

        before = len(model.classes)
        after = len(cache.cache[signals.route][cmask][2])
        rows_before += before
        rows_after += after
        mask_counts[cmask] += 1
        rr = routes[signals.route]
        rr["n"] += 1
        rr["rows_before"] += before
        rr["rows_after"] += after
        rr["base_hits"] += ah
        rr["screen_hits"] += bh

        if (a[0] if a else None) != (b[0] if b else None):
            changed += 1
            if bh and not ah:
                beneficial += 1
            elif ah and not bh:
                harmful += 1
            else:
                neutral += 1

    n = len(rows)
    return {
        "n": n,
        "mask_semantic_mismatch_n": mask_mismatch,
        "truth_eliminated_n": truth_eliminated,
        "baseline_hits": base_hits,
        "screened_hits": screen_hits,
        "baseline_top1": base_hits / max(1, n),
        "screened_top1": screen_hits / max(1, n),
        "changed_top1": changed,
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral": neutral,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "row_reduction_fraction": (rows_before - rows_after) / max(1, rows_before),
        "mask_counts": {str(k): v for k, v in sorted(mask_counts.items())},
        "routes": {
            r: {
                "n": c["n"],
                "row_reduction_fraction": (c["rows_before"] - c["rows_after"]) / max(1, c["rows_before"]),
                "delta_hits": c["screen_hits"] - c["base_hits"],
            }
            for r, c in routes.items()
        },
    }


def prepare(rows, bundle):
    prepared = []
    for s in rows:
        signals = _signals(s["data"])
        vec = baseline_from_array(hmt768_array(s["data"]))
        model = bundle.baseline.route_models[signals.route]
        prepared.append((s["data"], signals.route, vec, model))
    return prepared


def bench(prepared, bundle, cache, repeats=31):
    observed = {(route, h2f_structural_mask.invalid_mask(data)) for data, route, _, _ in prepared}
    for route, mask in observed:
        _ = cache.cache[route][mask]

    def measure(fn):
        vals = []
        sink = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            sink += fn()
            vals.append((time.perf_counter_ns() - t0) / max(1, len(prepared)))
        return {
            "median_ns_per_sample": statistics.median(vals),
            "min_ns_per_sample": min(vals),
            "max_ns_per_sample": max(vals),
            "sink": sink,
        }

    def baseline_score_only():
        sink = 0.0
        for _, _, vec, model in prepared:
            raw = _raw(model, vec)
            sink += float(raw[0])
        return sink

    def screened_score_only():
        sink = 0.0
        for data, route, vec, _ in prepared:
            mask = h2f_structural_mask.invalid_mask(data)
            raw, _ = cache.score(route, mask, vec)
            sink += float(raw[0])
        return sink

    def mask_only():
        sink = 0.0
        for data, _, _, _ in prepared:
            sink += h2f_structural_mask.invalid_mask(data)
        return sink

    def baseline_stage():
        sink = 0.0
        for data, _, _, _ in prepared:
            signals = _signals(data)
            vec = baseline_from_array(hmt768_array(data))
            model = bundle.baseline.route_models[signals.route]
            raw = _raw(model, vec)
            sink += float(raw[0])
        return sink

    def screened_stage():
        sink = 0.0
        for data, _, _, _ in prepared:
            signals = _signals(data)
            mask = h2f_structural_mask.invalid_mask(data)
            vec = baseline_from_array(hmt768_array(data))
            raw, _ = cache.score(signals.route, mask, vec)
            sink += float(raw[0])
        return sink

    baseline_score = measure(baseline_score_only)
    screened_score = measure(screened_score_only)
    mask_cost = measure(mask_only)
    baseline_total = measure(baseline_stage)
    screened_total = measure(screened_stage)

    return {
        "repeats": repeats,
        "mask_only": mask_cost,
        "scorer_only": {
            "baseline": baseline_score,
            "screened_including_mask": screened_score,
            "delta_ns": screened_score["median_ns_per_sample"] - baseline_score["median_ns_per_sample"],
            "delta_fraction": screened_score["median_ns_per_sample"] / baseline_score["median_ns_per_sample"] - 1.0,
        },
        "stage_end_to_end": {
            "baseline": baseline_total,
            "screened": screened_total,
            "delta_ns": screened_total["median_ns_per_sample"] - baseline_total["median_ns_per_sample"],
            "delta_fraction": screened_total["median_ns_per_sample"] / baseline_total["median_ns_per_sample"] - 1.0,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--expected-model-sha256", required=True)
    args = ap.parse_args()

    model_path = Path(args.model)
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual != args.expected_model_sha256:
        raise RuntimeError(f"model SHA mismatch: {actual}")

    bundle = load_bundle(model_path)
    cache = ScreenCache(bundle)
    all_rows = []
    per_crawl = {}
    collection = {}

    for crawl in ("CC-MAIN-2026-39", "CC-MAIN-2026-34"):
        rows, paths, stats = collect(crawl)
        rows = dedupe(rows)
        all_rows.extend(rows)
        per_crawl[crawl] = semantic_and_accuracy(rows, bundle, cache)
        collection[crawl] = {"n": len(rows), "warc_paths_considered": len(paths), "stats": stats}

    pooled = dedupe(all_rows)
    pooled_eval = semantic_and_accuracy(pooled, bundle, cache)
    prepared = prepare(pooled, bundle)
    timing = bench(prepared, bundle, cache)

    print(json.dumps({
        "phase": "h2f_production_shape_structural_screen",
        "policy": {
            "r22_used": False,
            "parameters_tuned": False,
            "retrained": False,
            "release_candidate_modified": False,
            "screen_semantics": "exact H1 permissive byte grammar",
            "screen_impl": "compiled Cython, bounded 4096 bytes",
            "score_impl": "precomputed route+mask contiguous weight submatrices",
            "mask_variants": 32,
            "downstream_not_integrated": True,
        },
        "collection": collection,
        "per_crawl": per_crawl,
        "pooled": pooled_eval,
        "timing": timing,
        "gate": {
            "semantic_equivalence_to_h1": pooled_eval["mask_semantic_mismatch_n"] == 0,
            "zero_truth_elimination": pooled_eval["truth_eliminated_n"] == 0,
            "zero_harm": pooled_eval["harmful"] == 0,
            "cross_crawl_nonnegative": all(v["screened_hits"] >= v["baseline_hits"] for v in per_crawl.values()),
            "net_stage_runtime_improves": timing["stage_end_to_end"]["delta_ns"] < 0,
        },
        "interpretation_rule": {
            "promote_architecture_if": "semantic/safety gates pass and stage end-to-end runtime improves",
            "reject_runtime_if": "screen cost exceeds scorer savings; do not tune thresholds to rescue it",
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
