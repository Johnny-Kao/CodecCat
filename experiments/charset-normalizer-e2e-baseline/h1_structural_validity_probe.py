from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from codeccat import Detector, load_bundle

import charset_external_scorer_domain_shift_ab as base

SAMPLE_BYTES = 4096
STRUCTURAL = ("big5", "gb18030", "shift-jis", "euc-jp", "euc-kr")


def _prefix(data: bytes) -> memoryview:
    return memoryview(data)[:SAMPLE_BYTES]


def valid_big5(data: bytes) -> bool:
    b = _prefix(data)
    i = 0
    n = len(b)
    while i < n:
        x = b[i]
        if x <= 0x7F:
            i += 1
            continue
        if not (0x81 <= x <= 0xFE):
            return False
        if i + 1 >= n:
            return True
        y = b[i + 1]
        if not (0x40 <= y <= 0x7E or 0xA1 <= y <= 0xFE):
            return False
        i += 2
    return True


def valid_gb18030(data: bytes) -> bool:
    b = _prefix(data)
    i = 0
    n = len(b)
    while i < n:
        x = b[i]
        if x <= 0x7F:
            i += 1
            continue
        if not (0x81 <= x <= 0xFE):
            return False
        if i + 1 >= n:
            return True
        y = b[i + 1]
        if 0x30 <= y <= 0x39:
            if i + 3 >= n:
                return True
            z = b[i + 2]
            w = b[i + 3]
            if not (0x81 <= z <= 0xFE and 0x30 <= w <= 0x39):
                return False
            i += 4
            continue
        if not (0x40 <= y <= 0x7E or 0x80 <= y <= 0xFE):
            return False
        i += 2
    return True


def valid_shift_jis(data: bytes) -> bool:
    b = _prefix(data)
    i = 0
    n = len(b)
    while i < n:
        x = b[i]
        if x <= 0x7F or 0xA1 <= x <= 0xDF:
            i += 1
            continue
        if not (0x81 <= x <= 0x9F or 0xE0 <= x <= 0xFC):
            return False
        if i + 1 >= n:
            return True
        y = b[i + 1]
        if not (0x40 <= y <= 0x7E or 0x80 <= y <= 0xFC):
            return False
        i += 2
    return True


def valid_euc_jp(data: bytes) -> bool:
    b = _prefix(data)
    i = 0
    n = len(b)
    while i < n:
        x = b[i]
        if x <= 0x7F:
            i += 1
            continue
        if x == 0x8E:
            if i + 1 >= n:
                return True
            if not (0xA1 <= b[i + 1] <= 0xDF):
                return False
            i += 2
            continue
        if x == 0x8F:
            if i + 2 >= n:
                return True
            if not (0xA1 <= b[i + 1] <= 0xFE and 0xA1 <= b[i + 2] <= 0xFE):
                return False
            i += 3
            continue
        if not (0xA1 <= x <= 0xFE):
            return False
        if i + 1 >= n:
            return True
        if not (0xA1 <= b[i + 1] <= 0xFE):
            return False
        i += 2
    return True


def valid_euc_kr(data: bytes) -> bool:
    b = _prefix(data)
    i = 0
    n = len(b)
    while i < n:
        x = b[i]
        if x <= 0x7F:
            i += 1
            continue
        if not (0xA1 <= x <= 0xFE):
            return False
        if i + 1 >= n:
            return True
        if not (0xA1 <= b[i + 1] <= 0xFE):
            return False
        i += 2
    return True


VALIDATORS = {
    "big5": valid_big5,
    "gb18030": valid_gb18030,
    "shift-jis": valid_shift_jis,
    "euc-jp": valid_euc_jp,
    "euc-kr": valid_euc_kr,
}


def normalize(label: str | None) -> str | None:
    return base.normalize_label(label)


def collect(crawl: str):
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


def dedupe(rows):
    seen = set()
    out = []
    for s in rows:
        key = (hashlib.sha256(s["data"]).digest(), normalize(s["label"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def analyze_crawl(crawl: str, rows, detector: Detector):
    truth = {k: Counter() for k in STRUCTURAL}
    candidate = Counter()
    invalid_by_label = Counter()
    top1_impossible_by_label = Counter()
    rescue_by_label = Counter()

    for s in rows:
        truth_label = normalize(s["label"])
        if truth_label in VALIDATORS:
            truth[truth_label]["n"] += 1
            if not VALIDATORS[truth_label](s["data"]):
                truth[truth_label]["invalid"] += 1

        rank = detector.rank(s["data"])
        top3 = rank[:3]
        for cand in top3:
            if cand not in VALIDATORS:
                continue
            candidate["covered_top3"] += 1
            if not VALIDATORS[cand](s["data"]):
                candidate["invalid_top3"] += 1
                invalid_by_label[cand] += 1
                if cand != truth_label:
                    candidate["wrong_invalid_top3"] += 1
                else:
                    candidate["truth_invalid_top3"] += 1

        if rank and rank[0] in VALIDATORS and not VALIDATORS[rank[0]](s["data"]):
            candidate["top1_impossible"] += 1
            top1_impossible_by_label[rank[0]] += 1
            next_valid = next(
                (
                    c for c in rank[1:]
                    if c not in VALIDATORS or VALIDATORS[c](s["data"])
                ),
                None,
            )
            if next_valid == truth_label:
                candidate["descriptive_rescue"] += 1
                rescue_by_label[truth_label] += 1

    truth_report = {}
    for label in STRUCTURAL:
        n = truth[label]["n"]
        inv = truth[label]["invalid"]
        truth_report[label] = {
            "n": n,
            "invalid": inv,
            "false_impossible_rate": inv / max(1, n),
        }

    return {
        "crawl": crawl,
        "n": len(rows),
        "truth_safety": truth_report,
        "candidate_elimination_descriptive": {
            **dict(candidate),
            "invalid_by_label": dict(invalid_by_label),
            "top1_impossible_by_label": dict(top1_impossible_by_label),
            "descriptive_rescue_by_truth": dict(rescue_by_label),
        },
    }


def timing(rows, detector: Detector, repeats=7):
    # Cost of evaluating only structurally constrained candidates in current top-3.
    ranks = [detector.rank(s["data"])[:3] for s in rows]
    vals = []
    sink = 0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for s, rank in zip(rows, ranks):
            for cand in rank:
                fn = VALIDATORS.get(cand)
                if fn is not None:
                    sink += int(fn(s["data"]))
        vals.append((time.perf_counter_ns() - start) / max(1, len(rows)))
    return {
        "median_ns_per_sample": statistics.median(vals),
        "min_ns_per_sample": min(vals),
        "max_ns_per_sample": max(vals),
        "repeats": repeats,
        "sink": sink,
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

    crawls = ["CC-MAIN-2026-39", "CC-MAIN-2026-34"]
    reports = []
    all_rows = []
    collection = {}
    for crawl in crawls:
        rows, paths, stats = collect(crawl)
        rows = dedupe(rows)
        all_rows.extend(rows)
        collection[crawl] = {
            "n": len(rows),
            "warc_paths_considered": len(paths),
            "stats": stats,
        }
        reports.append(analyze_crawl(crawl, rows, detector))

    pooled = dedupe(all_rows)
    pooled_report = analyze_crawl("pooled-development", pooled, detector)

    total_truth = Counter()
    total_invalid = Counter()
    for r in reports:
        for label, m in r["truth_safety"].items():
            total_truth[label] += m["n"]
            total_invalid[label] += m["invalid"]

    safety_pass = all(total_invalid[label] == 0 for label in STRUCTURAL if total_truth[label] > 0)

    print(json.dumps({
        "phase": "h1_structural_validity_probe",
        "policy": {
            "sample_bytes": SAMPLE_BYTES,
            "end_boundary_incomplete_sequence": "unknown/keep candidate",
            "elimination_semantics": "only explicit structural violation => impossible",
            "validators": list(STRUCTURAL),
            "r22_used_for_selection": False,
            "prediction_behavior_changed": False,
        },
        "collection": collection,
        "per_crawl": reports,
        "pooled": pooled_report,
        "safety_gate": {
            "pass_zero_false_impossible": safety_pass,
            "truth_n": dict(total_truth),
            "false_impossible_n": dict(total_invalid),
        },
        "timing_top3_structural_checks": timing(pooled, detector),
        "interpretation": {
            "H1_pass": "zero false-impossible on all observed covered ground-truth encodings; then advance to held-out H2 integration",
            "H1_fail": "do not integrate; inspect specification/validator breadth without using R22 errors",
            "candidate_elimination": "descriptive only; not a promotion criterion in H1",
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
