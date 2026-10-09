from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from codeccat import load_bundle
from codeccat.features import baseline_from_array, hmt768_array
from codeccat.runtime import _raw, _signals

import charset_external_scorer_domain_shift_ab as base
from h1_structural_validity_probe import VALIDATORS
from h2_fused_candidate_mask_ab import collect, dedupe, normalize


def possible(label: str, data: bytes) -> bool:
    fn = VALIDATORS.get(label)
    return True if fn is None else fn(data)


def rank_raw(model, vector):
    raw = _raw(model, vector)
    order = raw.argsort()[::-1]
    return tuple(np.asarray(model.classes, dtype=object)[order]), raw


def rank_screened(model, vector, data):
    raw = _raw(model, vector)
    keep = np.ones(len(model.classes), dtype=bool)
    removed = []
    for i, label in enumerate(model.classes):
        if label in VALIDATORS and not possible(label, data):
            keep[i] = False
            removed.append(label)
    idx = np.nonzero(keep)[0]
    order_local = raw[idx].argsort()[::-1]
    order = idx[order_local]
    return tuple(np.asarray(model.classes, dtype=object)[order]), raw, tuple(removed), int(keep.sum())


def evaluate(rows, bundle):
    base_hits = screened_hits = 0
    changed = beneficial = harmful = neutral = 0
    truth_eliminated = 0
    rows_before = rows_after = 0
    removed_by_label = Counter()
    route_stats = defaultdict(Counter)

    for s in rows:
        truth = normalize(s["label"])
        signals = _signals(s["data"])
        vector = baseline_from_array(hmt768_array(s["data"]))
        model = bundle.baseline.route_models[signals.route]

        baseline_rank, _ = rank_raw(model, vector)
        screened_rank, _, removed, kept = rank_screened(model, vector, s["data"])

        rows_before += len(model.classes)
        rows_after += kept
        for label in removed:
            removed_by_label[label] += 1
        if truth in removed:
            truth_eliminated += 1

        bh = int(bool(baseline_rank) and baseline_rank[0] == truth)
        sh = int(bool(screened_rank) and screened_rank[0] == truth)
        base_hits += bh
        screened_hits += sh

        rs = route_stats[signals.route]
        rs["n"] += 1
        rs["base_hits"] += bh
        rs["screened_hits"] += sh
        rs["rows_before"] += len(model.classes)
        rs["rows_after"] += kept

        if (baseline_rank[0] if baseline_rank else None) != (screened_rank[0] if screened_rank else None):
            changed += 1
            if sh and not bh:
                beneficial += 1
            elif bh and not sh:
                harmful += 1
            else:
                neutral += 1

    n = len(rows)
    return {
        "n": n,
        "baseline_hits": base_hits,
        "screened_hits": screened_hits,
        "baseline_top1": base_hits / max(1, n),
        "screened_top1": screened_hits / max(1, n),
        "delta_pp": 100.0 * (screened_hits - base_hits) / max(1, n),
        "changed_top1": changed,
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral": neutral,
        "truth_eliminated_n": truth_eliminated,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "row_reduction_n": rows_before - rows_after,
        "row_reduction_fraction": (rows_before - rows_after) / max(1, rows_before),
        "removed_by_label": dict(removed_by_label),
        "routes": {
            r: {
                "n": c["n"],
                "baseline_top1": c["base_hits"] / max(1, c["n"]),
                "screened_top1": c["screened_hits"] / max(1, c["n"]),
                "delta_hits": c["screened_hits"] - c["base_hits"],
                "row_reduction_fraction": (c["rows_before"] - c["rows_after"]) / max(1, c["rows_before"]),
            }
            for r, c in route_stats.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--expected-model-sha256", required=True)
    args = ap.parse_args()

    model = Path(args.model)
    actual = hashlib.sha256(model.read_bytes()).hexdigest()
    if actual != args.expected_model_sha256:
        raise RuntimeError(f"model SHA mismatch: {actual}")

    bundle = load_bundle(model)
    all_rows = []
    per_crawl = {}
    collection = {}

    for crawl in ("CC-MAIN-2026-39", "CC-MAIN-2026-34"):
        rows, paths, stats = collect(crawl)
        rows = dedupe(rows)
        all_rows.extend(rows)
        per_crawl[crawl] = evaluate(rows, bundle)
        collection[crawl] = {
            "n": len(rows),
            "warc_paths_considered": len(paths),
            "stats": stats,
        }

    pooled = dedupe(all_rows)
    pooled_eval = evaluate(pooled, bundle)

    print(json.dumps({
        "phase": "h2e_pre_score_structural_screen_ab",
        "policy": {
            "r22_used": False,
            "parameters_tuned": False,
            "retrained": False,
            "release_candidate_modified": False,
            "representation": "existing baseline 518-d",
            "route_models": "frozen R21 baseline route models",
            "screen": "H1 permissive spec-derived validators only",
            "downstream_calibrator_used": False,
            "question": "candidate-universe ordering effect on raw scorer",
        },
        "collection": collection,
        "per_crawl": per_crawl,
        "pooled": pooled_eval,
        "gate": {
            "zero_truth_elimination": pooled_eval["truth_eliminated_n"] == 0,
            "cross_crawl_nonnegative": all(v["screened_hits"] >= v["baseline_hits"] for v in per_crawl.values()),
            "nonzero_row_reduction": pooled_eval["row_reduction_n"] > 0,
            "no_parameter_adjustment": True,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
