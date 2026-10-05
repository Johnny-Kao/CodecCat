from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict

import chardet
import joblib
import numpy as np
from charset_normalizer import from_bytes

from codeccat import Detector, LinearModel, RuntimeBundle

import charset_external_scorer_domain_shift_ab as base
import charset_p2_multihop_kernel_tournament as p2


REPEATS = 7


def as_model(fused):
    if fused is None:
        return None
    classes, weight, bias = fused
    return LinearModel(
        classes=tuple(str(x) for x in classes),
        weight=np.asarray(weight, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
    )


def detector_from_fold(fd, classes, families):
    bundle = RuntimeBundle(
        route_models={
            route: as_model(p2.fuse_scaler_linear(model_tuple))
            for route, model_tuple in fd["models"].items()
        },
        calibrator=as_model(p2.fuse_estimator(fd["cal"])),
        triad=as_model(p2.fuse_estimator(fd["triad_cal"])),
        utf8_sig=as_model(p2.fuse_estimator(fd["sig_cal"])),
        utf8_gb=as_model(p2.fuse_estimator(fd["gb_cal"])),
        classes=tuple(str(x) for x in classes),
        families=tuple(str(x) for x in families),
    )
    return Detector(bundle)


def normalize(label):
    return base.normalize_label(label)


def cn_detect(data):
    match = from_bytes(data).best()
    return normalize(match.encoding if match is not None else None)


def chardet_detect(data):
    return normalize(chardet.detect(data).get("encoding"))


def bucket(length):
    if length <= 256:
        return "short<=256"
    if length <= 4096:
        return "medium<=4096"
    return "large>4096"


def timed_pass(rows, fn):
    sink = 0
    start = time.perf_counter_ns()
    for row in rows:
        pred = fn(row)
        sink += len(pred or "")
    elapsed = time.perf_counter_ns() - start
    return elapsed, sink


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    args = parser.parse_args()

    state = joblib.load(args.state)
    classes = state["classes"]
    families = state["families"]

    folds = []
    flat = []
    for fd in state["folds"]:
        detector = detector_from_fold(fd, classes, families)
        rows = fd["rows"]
        folds.append((fd["fold"], detector, rows))
        for row in rows:
            flat.append((fd["fold"], detector, row))

    assert len(flat) == 418

    def cc(item):
        _, detector, row = item
        ranked = detector.rank(row["data"])
        return normalize(ranked[0] if ranked else None)

    def cn(item):
        return cn_detect(item[2]["data"])

    def cd(item):
        return chardet_detect(item[2]["data"])

    systems = {
        "codeccat": cc,
        "charset-normalizer": cn,
        "chardet-7": cd,
    }

    accuracy = {}
    by_bucket = {}
    predictions = {}

    for name, fn in systems.items():
        hits = 0
        bucket_stats = defaultdict(Counter)
        pred_rows = []
        for item in flat:
            row = item[2]
            pred = fn(item)
            truth = normalize(row["label"])
            correct = pred == truth
            hits += int(correct)
            b = bucket(len(row["data"]))
            bucket_stats[b]["n"] += 1
            bucket_stats[b]["hits"] += int(correct)
            pred_rows.append(pred)
        accuracy[name] = {
            "n": len(flat),
            "hits": hits,
            "top1": hits / len(flat),
        }
        by_bucket[name] = {
            b: {
                "n": st["n"],
                "hits": st["hits"],
                "top1": st["hits"] / max(1, st["n"]),
            }
            for b, st in sorted(bucket_stats.items())
        }
        predictions[name] = pred_rows

    timing = {}
    for name, fn in systems.items():
        # warm-up
        timed_pass(flat, fn)
        samples = []
        sink = 0
        for _ in range(REPEATS):
            elapsed, current_sink = timed_pass(flat, fn)
            samples.append(elapsed / len(flat))
            sink += current_sink
        timing[name] = {
            "repeats": REPEATS,
            "median_ns_per_sample": statistics.median(samples),
            "min_ns_per_sample": min(samples),
            "max_ns_per_sample": max(samples),
            "sink": sink,
        }

    codeccat_ns = timing["codeccat"]["median_ns_per_sample"]
    for name, values in timing.items():
        values["relative_to_codeccat"] = values["median_ns_per_sample"] / codeccat_ns

    print(json.dumps({
        "phase": "formal_three_way_heldout_benchmark",
        "corpus_fingerprint": state["corpus_fingerprint"],
        "n": len(flat),
        "comparators": [
            "CodecCat clean package runtime",
            "charset-normalizer",
            "chardet 7",
        ],
        "accuracy": accuracy,
        "accuracy_by_input_size": by_bucket,
        "timing": timing,
        "methodology": {
            "model_construction_excluded": True,
            "same_process": True,
            "same_rows": True,
            "timing_repeats": REPEATS,
            "note": (
                "CodecCat numbers are cross-validated fold-fixture evidence. "
                "They are not release-model evidence."
            ),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
