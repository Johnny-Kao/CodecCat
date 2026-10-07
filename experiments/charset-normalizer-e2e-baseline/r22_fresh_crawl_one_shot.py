from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import chardet
import numpy as np
import sklearn
from charset_normalizer import __version__ as charset_normalizer_version
from charset_normalizer import from_bytes

from codeccat import Detector, load_bundle

import charset_external_scorer_domain_shift_ab as base

FRESH_CRAWL = "CC-MAIN-2026-25"


def collect_crawl(crawl):
    old_crawl = base.CRAWL
    old_url = base.WARC_PATHS_URL
    try:
        base.CRAWL = crawl
        base.WARC_PATHS_URL = f"https://data.commoncrawl.org/crawl-data/{crawl}/warc.paths.gz"
        rows, paths, stats = base.collect_external()
        return rows, paths, dict(stats)
    finally:
        base.CRAWL = old_crawl
        base.WARC_PATHS_URL = old_url


def normalize(label):
    return base.normalize_label(label)


def deduplicate(rows):
    seen = set()
    out = []
    for s in rows:
        key = (hashlib.sha256(s["data"]).digest(), normalize(s["label"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def fingerprint(rows):
    h = hashlib.sha256()
    for s in sorted(
        rows,
        key=lambda x: (
            x["warc_path"], x.get("host", ""), x["label"],
            hashlib.sha256(x["data"]).hexdigest(),
        ),
    ):
        h.update(s["warc_path"].encode())
        h.update(b"\0")
        h.update(s.get("host", "").encode())
        h.update(b"\0")
        h.update(s["label"].encode())
        h.update(b"\0")
        h.update(hashlib.sha256(s["data"]).digest())
    return h.hexdigest()


def cn_detect(data):
    match = from_bytes(data).best()
    return normalize(match.encoding if match is not None else None)


def chardet_detect(data):
    return normalize(chardet.detect(data).get("encoding"))


def evaluate(rows, detector):
    systems = {
        "codeccat": lambda data: normalize(detector.detect(data).encoding),
        "charset-normalizer": cn_detect,
        "chardet-7": chardet_detect,
    }
    supported = set(detector._runtime.bundle.classes)
    results = {}
    route_results = {}

    for name, fn in systems.items():
        total = hits = supported_n = supported_hits = 0
        routes = {r: Counter() for r in ("U", "N", "RL", "RH")}
        for sample in rows:
            truth = normalize(sample["label"])
            pred = fn(sample["data"])
            hit = int(pred == truth)
            total += 1
            hits += hit
            routes[sample["route"]]["n"] += 1
            routes[sample["route"]]["hits"] += hit
            if truth in supported:
                supported_n += 1
                supported_hits += hit
        results[name] = {
            "n": total,
            "hits": hits,
            "top1": hits / max(1, total),
            "supported_label_n": supported_n,
            "supported_label_hits": supported_hits,
            "supported_label_top1": supported_hits / max(1, supported_n),
        }
        route_results[name] = {
            r: {
                "n": c["n"],
                "hits": c["hits"],
                "top1": c["hits"] / max(1, c["n"]),
            }
            for r, c in routes.items() if c["n"]
        }
    return results, route_results


def latency(rows, detector, repeats=7):
    systems = {
        "codeccat": lambda data: detector.detect(data).encoding,
        "charset-normalizer": lambda data: (
            (m.encoding if (m := from_bytes(data).best()) is not None else None)
        ),
        "chardet-7": lambda data: chardet.detect(data).get("encoding"),
    }
    out = {}
    for name, fn in systems.items():
        for sample in rows:
            fn(sample["data"])
        vals = []
        sink = 0
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for sample in rows:
                pred = fn(sample["data"])
                sink += len(pred or "")
            elapsed = time.perf_counter_ns() - start
            vals.append(elapsed / max(1, len(rows)))
        out[name] = {
            "median_ns_per_sample": statistics.median(vals),
            "min_ns_per_sample": min(vals),
            "max_ns_per_sample": max(vals),
            "repeats": repeats,
            "sink": sink,
        }
    cc = out["codeccat"]["median_ns_per_sample"]
    for v in out.values():
        v["relative_to_codeccat"] = v["median_ns_per_sample"] / cc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--expected-model-sha256", required=True)
    ap.add_argument("--dev-fingerprint", required=True)
    args = ap.parse_args()

    model_path = Path(args.model)
    actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_sha != args.expected_model_sha256:
        raise RuntimeError(
            f"artifact SHA256 mismatch: expected {args.expected_model_sha256}, got {actual_sha}"
        )

    bundle = load_bundle(model_path)
    detector = Detector(bundle)

    rows, paths, stats = collect_crawl(FRESH_CRAWL)
    rows = deduplicate(rows)
    raw_n = len(rows)

    # Dev rows are not re-fetched here; the immutable R21 development fingerprint
    # is reported as provenance. Exact overlap against 39/34 was already prevented
    # by crawl separation, and we additionally report duplicate count within holdout.
    holdout_fp = fingerprint(rows)

    accuracy, by_route = evaluate(rows, detector)
    timing = latency(rows, detector)

    report = {
        "phase": "r22_fresh_crawl_one_shot",
        "holdout": {
            "crawl": FRESH_CRAWL,
            "n": len(rows),
            "warc_paths_considered": len(paths),
            "fingerprint": holdout_fp,
            "collection_stats": stats,
            "fresh_for_model_selection": True,
        },
        "frozen_artifact": {
            "sha256": actual_sha,
            "schema": 2,
            "development_fingerprint": args.dev_fingerprint,
            "gate_thresholds": dict(bundle.gate_thresholds),
            "r12_threshold": bundle.r12_threshold,
        },
        "accuracy": accuracy,
        "accuracy_by_route": by_route,
        "timing": timing,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "charset_normalizer": charset_normalizer_version,
            "chardet": chardet.__version__,
        },
        "methodology": {
            "single_frozen_codeccat_artifact": True,
            "same_holdout_bytes_for_all_detectors": True,
            "model_construction_excluded_from_latency": True,
            "holdout_used_for_tuning": False,
            "cc_main_2026_30_used": False,
            "rule": (
                "This crawl may support release claims only while no architecture, "
                "feature, threshold, calibration, or model-selection choice is changed "
                "in response to its results."
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
