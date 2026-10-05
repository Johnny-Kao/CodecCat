from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import joblib
import numpy as np
from pathlib import Path

import charset_external_scorer_domain_shift_ab as base
import charset_training_tournament as trainmod
from charset_cost_aware_routing_tree import collect as collect_legacy


CANDIDATES = (
    "hmt768",
    "hmt1536",
    "hmt3072",
    "prefix4096",
    "hmt4096",
)


def hmt(data: bytes, total: int) -> bytes:
    if len(data) <= total:
        return data
    # Split budget approximately 1/3 head, middle, tail.
    head = total // 3
    middle = total - 2 * head
    center = len(data) // 2
    left = center - middle // 2
    right = left + middle
    return data[:head] + data[left:right] + data[-head:]


def sample_bytes(data: bytes, mode: str) -> bytes:
    if mode == "hmt768":
        return hmt(data, 768)
    if mode == "hmt1536":
        return hmt(data, 1536)
    if mode == "hmt3072":
        return hmt(data, 3072)
    if mode == "prefix4096":
        return data[:4096]
    if mode == "hmt4096":
        return hmt(data, 4096)
    raise ValueError(mode)


def features(data: bytes, mode: str) -> np.ndarray:
    data = sample_bytes(data, mode)
    a = np.frombuffer(data, dtype=np.uint8)
    n = max(1, len(a))

    unigram = np.bincount(a, minlength=256).astype(np.float32) / n

    bigram = np.zeros(256, dtype=np.float32)
    if len(a) >= 2:
        # Match locked wrapped uint8 adjacent-byte-sum representation.
        bins = a[:-1] + a[1:]
        bigram = np.bincount(bins, minlength=256).astype(np.float32) / (len(a) - 1)

    scalars = np.array(
        [
            math.log2(n + 1) / 16.0,
            float(np.mean(a >= 128)) if len(a) else 0.0,
            float(np.mean(a == 0)) if len(a) else 0.0,
            float(np.mean((a >= 32) & (a <= 126))) if len(a) else 0.0,
            float(np.mean(a == 10)) if len(a) else 0.0,
            float(np.mean(a == 13)) if len(a) else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([unigram, bigram, scalars])


def normalize_legacy_label(row):
    raw = base.normalize_label(base.canonical_encoding(row["encoding"])) or row["encoding"]
    return raw


def legacy_rl_rows():
    rows, _ = collect_legacy()
    out = []
    for row in rows:
        data = row["path"].read_bytes()
        if base.route_bucket(data) != "RL":
            continue
        out.append({"data": data, "label": normalize_legacy_label(row)})
    return out


def canonical_development_rows(state):
    seen = set()
    rows = []
    for fd in state["folds"]:
        for sample in fd["rows"]:
            if sample["route"] != "RL":
                continue
            key = (sample["warc_path"], sample.get("host", ""), sample["label"], sample["data"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(sample)
    return rows


def fit_candidate(train_rows, mode):
    X = np.stack([features(row["data"], mode) for row in train_rows])
    y = np.asarray([row["label"] for row in train_rows], dtype=object)
    return trainmod.fit_linear(X, y, C=0.5)


def rank(model_tuple, data, mode):
    scaler, model = model_tuple
    x = features(data, mode)
    scores = model.decision_function(scaler.transform(x[None, :]))
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    order = np.argsort(scores[0])[::-1]
    return [str(x) for x in model.classes_[order]]


def evaluate(model_tuple, rows, mode):
    hits = Counter()
    truth_rank = Counter()
    confusions = Counter()
    for row in rows:
        ranked = rank(model_tuple, row["data"], mode)
        truth = row["label"]
        try:
            pos = ranked.index(truth) + 1
        except ValueError:
            pos = 999
        hits["top1"] += int(pos == 1)
        hits["top3"] += int(pos <= 3)
        hits["top5"] += int(pos <= 5)
        if pos != 1:
            confusions[(truth, ranked[0] if ranked else None)] += 1
        truth_rank[str(pos if pos < 999 else "absent")] += 1
    n = len(rows)
    return {
        "n": n,
        "hits": dict(hits),
        "top1": hits["top1"] / n if n else 0.0,
        "top3": hits["top3"] / n if n else 0.0,
        "top5": hits["top5"] / n if n else 0.0,
        "truth_rank": dict(truth_rank),
        "top_confusions": [
            {"truth": t, "pred": p, "n": count}
            for (t, p), count in confusions.most_common(12)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--development-crawl", default="CC-MAIN-2026-34")
    parser.add_argument("--development-corpus-cache", required=True)
    args = parser.parse_args()

    state = joblib.load(args.state)
    legacy = legacy_rl_rows()
    canonical = canonical_development_rows(state)
    cache_path = Path(args.development_corpus_cache)
    if cache_path.exists():
        cached = joblib.load(cache_path)
        if cached.get("schema") != 1 or cached.get("crawl") != args.development_crawl:
            raise RuntimeError("R7 development corpus cache metadata mismatch")
        test_rl = cached["rows"]
        dev_stats = cached["collection_stats"]
        corpus_source = "cache-hit"
    else:
        # Freeze one expanded RL development sample. Live Common Crawl range
        # availability can vary between runs, so architecture selection must
        # not silently compare different byte sets.
        old = {
            "CRAWL": base.CRAWL,
            "WARC_PATHS_URL": base.WARC_PATHS_URL,
            "N_WARC_FILES": base.N_WARC_FILES,
            "MAX_ACCEPTED": base.MAX_ACCEPTED,
            "MAX_ROUTE": dict(base.MAX_ROUTE),
            "TARGET_RESIDUAL": base.TARGET_RESIDUAL,
        }
        try:
            base.CRAWL = args.development_crawl
            base.WARC_PATHS_URL = f"https://data.commoncrawl.org/crawl-data/{args.development_crawl}/warc.paths.gz"
            base.N_WARC_FILES = 64
            base.MAX_ACCEPTED = 640
            base.MAX_ROUTE = {"U": 160, "N": 80, "R": 440}
            base.TARGET_RESIDUAL = 360
            dev_rows, _, stats = base.collect_external()
            test_rl = [row for row in dev_rows if row["route"] == "RL"]
            dev_stats = dict(stats)
        finally:
            base.CRAWL = old["CRAWL"]
            base.WARC_PATHS_URL = old["WARC_PATHS_URL"]
            base.N_WARC_FILES = old["N_WARC_FILES"]
            base.MAX_ACCEPTED = old["MAX_ACCEPTED"]
            base.MAX_ROUTE = old["MAX_ROUTE"]
            base.TARGET_RESIDUAL = old["TARGET_RESIDUAL"]

        if len(test_rl) < 50:
            raise RuntimeError(
                f"expanded R7 development collection produced only {len(test_rl)} RL rows; "
                "refuse to freeze an undersized architecture-selection corpus"
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "schema": 1,
                "crawl": args.development_crawl,
                "rows": test_rl,
                "collection_stats": dev_stats,
            },
            cache_path,
            compress=3,
        )
        corpus_source = "newly-frozen"
    known = set(row["label"] for row in legacy)
    train_rows = legacy + [row for row in canonical if row["label"] in known]

    results = {}
    for mode in CANDIDATES:
        model = fit_candidate(train_rows, mode)
        results[mode] = evaluate(model, test_rl, mode)

    baseline = results["hmt768"]["top1"]
    for mode, result in results.items():
        result["delta_top1_pp_vs_hmt768"] = (result["top1"] - baseline) * 100.0
        result["extra_sample_budget_bytes"] = {
            "hmt768": 0,
            "hmt1536": 768,
            "hmt3072": 2304,
            "prefix4096": 3328,
            "hmt4096": 3328,
        }[mode]

    print(json.dumps({
        "phase": "r7_rl_sampling_tournament",
        "training": {
            "legacy_rl_rows": len(legacy),
            "canonical_rl_rows": len(canonical),
            "combined_training_rows": len(train_rows),
            "note": "Only canonical CC-MAIN-2026-39 plus legacy training are used to fit candidates.",
        },
        "development_test": {
            "crawl": args.development_crawl,
            "rl_rows": len(test_rl),
            "corpus_source": corpus_source,
            "corpus_cache": str(cache_path),
            "collection_stats": dict(dev_stats),
            "release_holdout_used": False,
            "cc_main_2026_30_used": False,
        },
        "candidates": results,
        "selection_rule": (
            "Advance only if an RL-only sampling candidate materially improves Top-1 "
            "and does not reduce Top-3/Top-5 retrieval; downstream reconstruction is a separate gate."
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
