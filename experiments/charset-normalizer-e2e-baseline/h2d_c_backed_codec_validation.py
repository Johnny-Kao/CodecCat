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

CODEC_MAP = {
    "big5": "big5",
    "gb18030": "gb18030",
    "shift-jis": "shift_jis",
    "euc-jp": "euc_jp",
    "euc-kr": "euc_kr",
}
SAMPLE_BYTES = 4096


def py_struct_valid(label: str, data: bytes) -> bool:
    fn = VALIDATORS.get(label)
    return True if fn is None else fn(data)


def codec_valid(label: str, data: bytes) -> bool:
    codec = CODEC_MAP.get(label)
    if codec is None:
        return True
    sample = data[:SAMPLE_BYTES]
    try:
        sample.decode(codec, "strict")
        return True
    except UnicodeDecodeError as e:
        # If the only failure is an incomplete multibyte sequence at the end of
        # the bounded observation window, treat it as unknown/possible.
        reason = (e.reason or "").lower()
        if e.end == len(sample) and ("incomplete" in reason or "truncated" in reason or "unexpected end" in reason):
            return True
        return False


def evaluate_semantics(rows):
    covered = mismatch = false_impossible_codec = false_possible_codec = 0
    by_label = Counter()
    truth_by_label = Counter()
    truth_false_impossible = Counter()

    for s in rows:
        truth = normalize(s["label"])
        for label in CODEC_MAP:
            # Compare semantics on top-level candidate validity for every row.
            a = py_struct_valid(label, s["data"])
            b = codec_valid(label, s["data"])
            covered += 1
            if a != b:
                mismatch += 1
                by_label[(label, "py_valid" if a else "py_invalid", "codec_valid" if b else "codec_invalid")] += 1

        if truth in CODEC_MAP:
            truth_by_label[truth] += 1
            if not codec_valid(truth, s["data"]):
                false_impossible_codec += 1
                truth_false_impossible[truth] += 1

    return {
        "candidate_checks_n": covered,
        "semantic_mismatch_n": mismatch,
        "semantic_mismatch_breakdown": {"|".join(k): v for k, v in by_label.items()},
        "truth_n": dict(truth_by_label),
        "truth_false_impossible_n": false_impossible_codec,
        "truth_false_impossible_by_label": dict(truth_false_impossible),
    }


def evaluate_top1(rows, detector):
    impossible_py = impossible_codec = 0
    changed_if_codec_veto = beneficial = harmful = neutral = 0

    for s in rows:
        truth = normalize(s["label"])
        rank = detector.rank(s["data"])
        if not rank:
            continue
        top = rank[0]
        if top in CODEC_MAP and not py_struct_valid(top, s["data"]):
            impossible_py += 1
        if top in CODEC_MAP and not codec_valid(top, s["data"]):
            impossible_codec += 1
            fallback = next((c for c in rank[1:] if codec_valid(c, s["data"])), top)
            if fallback != top:
                changed_if_codec_veto += 1
                before = int(top == truth)
                after = int(fallback == truth)
                if after and not before:
                    beneficial += 1
                elif before and not after:
                    harmful += 1
                else:
                    neutral += 1

    return {
        "impossible_top1_python_state_machine": impossible_py,
        "impossible_top1_codec_strict": impossible_codec,
        "changed_if_codec_veto": changed_if_codec_veto,
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral": neutral,
    }


def timing(rows, ranks, repeats=15):
    covered = [(s, r[0]) for s, r in zip(rows, ranks) if r and r[0] in CODEC_MAP]

    def bench(fn):
        vals = []
        sink = 0
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for s, label in covered:
                sink += int(fn(label, s["data"]))
            vals.append((time.perf_counter_ns() - start) / max(1, len(covered)))
        return {
            "n": len(covered),
            "median_ns_per_covered_sample": statistics.median(vals),
            "min_ns_per_covered_sample": min(vals),
            "max_ns_per_covered_sample": max(vals),
            "sink": sink,
        }

    return {
        "python_state_machine": bench(py_struct_valid),
        "codec_strict_decode": bench(codec_valid),
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
    all_rows = []
    per_crawl = {}
    collection = {}

    for crawl in ("CC-MAIN-2026-39", "CC-MAIN-2026-34"):
        rows, paths, stats = collect(crawl)
        rows = dedupe(rows)
        all_rows.extend(rows)
        per_crawl[crawl] = {
            "semantics": evaluate_semantics(rows),
            "top1": evaluate_top1(rows, detector),
        }
        collection[crawl] = {"n": len(rows), "warc_paths_considered": len(paths), "stats": stats}

    pooled = dedupe(all_rows)
    ranks = [detector.rank(s["data"]) for s in pooled]

    print(json.dumps({
        "phase": "h2d_c_backed_codec_validation",
        "policy": {
            "r22_used": False,
            "parameters_tuned": False,
            "release_candidate_modified": False,
            "candidate_scope": list(CODEC_MAP),
            "sample_bytes": SAMPLE_BYTES,
            "terminal_incomplete_window": "keep candidate",
        },
        "collection": collection,
        "per_crawl": per_crawl,
        "pooled": {
            "semantics": evaluate_semantics(pooled),
            "top1": evaluate_top1(pooled, detector),
        },
        "timing": timing(pooled, ranks),
        "decision_rule": {
            "required": [
                "zero truth false-impossible",
                "no behavioral harm",
                "materially lower cost than Python state machine",
            ],
            "note": "Semantic mismatch versus permissive structural DFA is acceptable only if codec strictness can be justified as true encoding invalidity rather than mapping-table coverage differences."
        }
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
