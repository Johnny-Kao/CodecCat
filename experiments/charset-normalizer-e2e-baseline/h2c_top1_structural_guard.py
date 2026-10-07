from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from codeccat import Detector, load_bundle

import charset_external_scorer_domain_shift_ab as base
from h1_structural_validity_probe import VALIDATORS
from h2_fused_candidate_mask_ab import normalize, collect, dedupe


def candidate_valid(label: str, data: bytes) -> bool:
    fn = VALIDATORS.get(label)
    return True if fn is None else fn(data)


def late_candidate_veto(rank, data):
    if not rank:
        return rank
    if candidate_valid(rank[0], data):
        return rank
    for cand in rank[1:]:
        if candidate_valid(cand, data):
            return (cand,) + tuple(x for x in rank if x != cand)
    return rank


def evaluate(rows, detector):
    base_hits = veto_hits = 0
    changed = beneficial = harmful = neutral = 0
    impossible_top1 = 0
    by_label = Counter()

    ranks = []
    for s in rows:
        rank = detector.rank(s["data"])
        ranks.append(rank)
        truth = normalize(s["label"])
        if rank and not candidate_valid(rank[0], s["data"]):
            impossible_top1 += 1
            by_label[rank[0]] += 1
        vrank = late_candidate_veto(rank, s["data"])
        bh = int(bool(rank) and rank[0] == truth)
        vh = int(bool(vrank) and vrank[0] == truth)
        base_hits += bh
        veto_hits += vh
        if (rank[0] if rank else None) != (vrank[0] if vrank else None):
            changed += 1
            if vh and not bh:
                beneficial += 1
            elif bh and not vh:
                harmful += 1
            else:
                neutral += 1

    return {
        "n": len(rows),
        "baseline_hits": base_hits,
        "veto_hits": veto_hits,
        "baseline_top1": base_hits / max(1, len(rows)),
        "veto_top1": veto_hits / max(1, len(rows)),
        "changed_top1": changed,
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral": neutral,
        "impossible_top1_n": impossible_top1,
        "impossible_top1_by_label": dict(by_label),
        "_ranks": ranks,
    }


def timing(rows, ranks, repeats=11):
    all_vals = []
    covered_vals = []
    all_sink = 0
    covered_sink = 0
    covered_n = sum(1 for rank in ranks if rank and rank[0] in VALIDATORS)

    for _ in range(repeats):
        start = time.perf_counter_ns()
        for s, rank in zip(rows, ranks):
            if rank:
                all_sink += int(candidate_valid(rank[0], s["data"]))
        all_vals.append((time.perf_counter_ns() - start) / max(1, len(rows)))

        start = time.perf_counter_ns()
        for s, rank in zip(rows, ranks):
            if rank and rank[0] in VALIDATORS:
                covered_sink += int(candidate_valid(rank[0], s["data"]))
        covered_vals.append((time.perf_counter_ns() - start) / max(1, covered_n))

    return {
        "all_samples": {
            "median_ns_per_sample": statistics.median(all_vals),
            "min_ns_per_sample": min(all_vals),
            "max_ns_per_sample": max(all_vals),
            "sink": all_sink,
        },
        "structural_top1_only": {
            "n": covered_n,
            "median_ns_per_covered_sample": statistics.median(covered_vals),
            "min_ns_per_covered_sample": min(covered_vals),
            "max_ns_per_covered_sample": max(covered_vals),
            "sink": covered_sink,
        },
        "repeats": repeats,
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

    detector = Detector(load_bundle(model))
    crawls = ("CC-MAIN-2026-39", "CC-MAIN-2026-34")
    all_rows = []
    per_crawl = {}
    collection = {}

    for crawl in crawls:
        rows, paths, stats = collect(crawl)
        rows = dedupe(rows)
        all_rows.extend(rows)
        ev = evaluate(rows, detector)
        ev.pop("_ranks")
        per_crawl[crawl] = ev
        collection[crawl] = {"n": len(rows), "warc_paths_considered": len(paths), "stats": stats}

    pooled = dedupe(all_rows)
    ev = evaluate(pooled, detector)
    ranks = ev.pop("_ranks")

    print(json.dumps({
        "phase": "h2c_top1_only_structural_guard",
        "policy": {
            "r22_used": False,
            "parameters_tuned": False,
            "release_candidate_modified": False,
            "validation_scope": "only final top1 candidate",
            "fallback": "next ranked structurally possible candidate",
            "learned_score_geometry_modified": False,
        },
        "collection": collection,
        "per_crawl": per_crawl,
        "pooled": ev,
        "timing": timing(pooled, ranks),
        "decision_rule": {
            "production_candidate_only_if": "near-zero behavioral risk and sufficiently small runtime cost",
            "accuracy_gain_required": False,
            "benchmark_fitting_forbidden": True,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
