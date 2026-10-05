from __future__ import annotations

# Holdout execution is rooted at the repository root.

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import chardet
import joblib
import numpy as np
import sklearn
from charset_normalizer import __version__ as charset_normalizer_version
from charset_normalizer import from_bytes

from codeccat import Detector, LinearModel, RuntimeBundle, save_bundle

import charset_candidate_calibration_ab as calmod
import charset_canonical_guarded_final_validation as canon
import charset_external_scorer_domain_shift_ab as base
import charset_external_scorer_learning_curve as lc
import charset_triad_specialist_ab as tri
import charset_p2_multihop_kernel_tournament as p2


def as_model(fused):
    if fused is None:
        return None
    classes, weight, bias = fused
    return LinearModel(
        classes=tuple(str(x) for x in classes),
        weight=np.asarray(weight, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
    )


def unique_development_rows(state):
    seen = set()
    rows = []
    for fd in state["folds"]:
        for sample in fd["rows"]:
            key = (
                sample["warc_path"],
                sample.get("host", ""),
                sample["label"],
                hashlib.sha256(sample["data"]).digest(),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(sample)
    return rows


def build_release_bundle(dev_rows):
    legacy_X, legacy_y, legacy_b = base.load_legacy_training()
    classes, families = calmod.build_vocab(legacy_y)

    models = lc.fit_route_models(dev_rows, legacy_X, legacy_y, legacy_b)
    cal = calmod.crossfit_calibration_rows(
        dev_rows, legacy_X, legacy_y, legacy_b, classes, families
    )
    triad_cal = tri.crossfit_triad_rows(
        dev_rows, legacy_X, legacy_y, legacy_b, classes, families
    )
    sig_cal = canon.fit_one(
        canon.SIG_PAIR, dev_rows, legacy_X, legacy_y, legacy_b, classes, families
    )
    gb_cal = canon.fit_one(
        canon.GB_PAIR, dev_rows, legacy_X, legacy_y, legacy_b, classes, families
    )

    required = {"U", "N", "RL", "RH"}
    if set(models) != required:
        raise RuntimeError(f"release model missing routes: {sorted(required - set(models))}")

    return RuntimeBundle(
        route_models={
            route: as_model(p2.fuse_scaler_linear(model_tuple))
            for route, model_tuple in models.items()
        },
        calibrator=as_model(p2.fuse_estimator(cal)),
        triad=as_model(p2.fuse_estimator(triad_cal)),
        utf8_sig=as_model(p2.fuse_estimator(sig_cal)),
        utf8_gb=as_model(p2.fuse_estimator(gb_cal)),
        classes=tuple(str(x) for x in classes),
        families=tuple(str(x) for x in families),
    )


def collect_holdout(crawl):
    old_crawl = base.CRAWL
    old_url = base.WARC_PATHS_URL
    try:
        base.CRAWL = crawl
        base.WARC_PATHS_URL = (
            f"https://data.commoncrawl.org/crawl-data/{crawl}/warc.paths.gz"
        )
        rows, paths, stats = base.collect_external()
        return rows, paths, stats
    finally:
        base.CRAWL = old_crawl
        base.WARC_PATHS_URL = old_url


def normalize(label):
    return base.normalize_label(label)


def fingerprint(rows):
    h = hashlib.sha256()
    for sample in sorted(
        rows,
        key=lambda x: (
            x["warc_path"],
            x.get("host", ""),
            x["label"],
            len(x["data"]),
        ),
    ):
        h.update(sample["warc_path"].encode())
        h.update(b"\0")
        h.update(sample.get("host", "").encode())
        h.update(b"\0")
        h.update(sample["label"].encode())
        h.update(b"\0")
        h.update(hashlib.sha256(sample["data"]).digest())
    return h.hexdigest()


def deduplicate_rows(rows):
    seen = set()
    out = []
    for sample in rows:
        digest = hashlib.sha256(sample["data"]).digest()
        key = (digest, normalize(sample["label"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(sample)
    return out


def exact_data_hashes(rows):
    return {hashlib.sha256(sample["data"]).digest() for sample in rows}


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
    for name, fn in systems.items():
        total = hits = supported_n = supported_hits = 0
        for sample in rows:
            truth = normalize(sample["label"])
            pred = fn(sample["data"])
            total += 1
            hits += int(pred == truth)
            if truth in supported:
                supported_n += 1
                supported_hits += int(pred == truth)
        results[name] = {
            "n": total,
            "hits": hits,
            "top1": hits / max(1, total),
            "supported_label_n": supported_n,
            "supported_label_hits": supported_hits,
            "supported_label_top1": supported_hits / max(1, supported_n),
        }
    return results


def latency(rows, detector, repeats=5):
    systems = {
        "codeccat": lambda data: detector.detect(data).encoding,
        "charset-normalizer": lambda data: (
            (m.encoding if (m := from_bytes(data).best()) is not None else None)
        ),
        "chardet-7": lambda data: chardet.detect(data).get("encoding"),
    }
    out = {}
    for name, fn in systems.items():
        # warmup
        for sample in rows:
            fn(sample["data"])
        samples = []
        sink = 0
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for sample in rows:
                pred = fn(sample["data"])
                sink += len(pred or "")
            elapsed = time.perf_counter_ns() - start
            samples.append(elapsed / max(1, len(rows)))
        out[name] = {
            "median_ns_per_sample": statistics.median(samples),
            "min_ns_per_sample": min(samples),
            "max_ns_per_sample": max(samples),
            "repeats": repeats,
            "sink": sink,
        }
    baseline = out["codeccat"]["median_ns_per_sample"]
    for values in out.values():
        values["relative_to_codeccat"] = values["median_ns_per_sample"] / baseline
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--holdout-crawl", required=True)
    parser.add_argument("--additional-development-crawl")
    args = parser.parse_args()

    state = joblib.load(args.state)
    canonical_dev_rows = unique_development_rows(state)
    additional_dev_rows = []
    additional_dev_stats = {}
    if args.additional_development_crawl:
        additional_dev_rows, _, stats = collect_holdout(args.additional_development_crawl)
        additional_dev_stats = dict(stats)

    dev_rows = deduplicate_rows(canonical_dev_rows + additional_dev_rows)
    dev_fp = fingerprint(dev_rows)

    started = time.perf_counter()
    bundle = build_release_bundle(dev_rows)
    training_seconds = time.perf_counter() - started

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(model_path, bundle)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    detector = Detector(bundle)

    holdout_rows, holdout_paths, holdout_stats = collect_holdout(args.holdout_crawl)

    dev_hashes = exact_data_hashes(dev_rows)
    exact_overlap_n = sum(
        hashlib.sha256(sample["data"]).digest() in dev_hashes
        for sample in holdout_rows
    )
    holdout_rows = [
        sample for sample in holdout_rows
        if hashlib.sha256(sample["data"]).digest() not in dev_hashes
    ]
    holdout_fp = fingerprint(holdout_rows)

    accuracy = evaluate(holdout_rows, detector)
    timing = latency(holdout_rows, detector)

    report = {
        "phase": "release_candidate_independent_holdout_v1",
        "frozen_architecture": {
            "runtime": "locked clean P15-equivalent package",
            "s3": 0.02,
            "triad": ["utf-8", "cp1251", "gb18030"],
            "pairs": [["utf-8", "utf-8-sig"], ["utf-8", "gb18030"]],
            "replacement_rate_guard": 0.02,
        },
        "development": {
            "canonical_source_crawl": "CC-MAIN-2026-39",
            "canonical_rows": len(canonical_dev_rows),
            "additional_source_crawl": args.additional_development_crawl,
            "additional_rows_collected": len(additional_dev_rows),
            "combined_unique_rows": len(dev_rows),
            "additional_collection_stats": additional_dev_stats,
            "fingerprint": dev_fp,
            "note": "No current independent-holdout row is used for fitting.",
        },
        "release_model": {
            "path": str(model_path),
            "sha256": model_sha256,
            "bytes": model_path.stat().st_size,
            "training_seconds": training_seconds,
        },
        "independent_holdout": {
            "crawl": args.holdout_crawl,
            "n": len(holdout_rows),
            "fingerprint": holdout_fp,
            "warc_paths_considered": len(holdout_paths),
            "exact_data_overlap_removed": exact_overlap_n,
            "collection_stats": dict(holdout_stats),
        },
        "accuracy": accuracy,
        "timing": timing,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "charset_normalizer": charset_normalizer_version,
            "chardet": chardet.__version__,
        },
        "methodology": {
            "holdout_used_for_tuning": False,
            "single_frozen_codeccat_model": True,
            "same_holdout_bytes_for_all_detectors": True,
            "model_construction_excluded_from_latency": True,
            "rule": (
                "If this holdout is used to change architecture, thresholds, "
                "features, calibration, or model-selection choices, it becomes "
                "development data and cannot support the release claim."
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
