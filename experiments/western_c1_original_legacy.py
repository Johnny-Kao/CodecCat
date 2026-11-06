"""Paired counterfactual on pre-examined original BYVoid/uchardet test files.

Not independent evidence: some fixtures overlap training data, and previous
source-level triage inspected them. This is a regression guard only.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import importlib.metadata
import json
from pathlib import Path
import sys

import codeccat
from western_c1_counterfactual import rerank_control

import uchardet_blind_holdout as source

OUTPUT = Path("research-results/western-c1-original-legacy-check.json")


def main():
    assert importlib.metadata.version("codeccat") == "0.1.0"
    rows, skips = source.source_samples()
    assert len(rows) == 102, len(rows)
    det = codeccat.CodecCat()
    outcomes = defaultdict(Counter)
    change_records = []
    for r in rows:
        found = det.detect(r["raw"])
        proposal, route = rerank_control(r["raw"], found)
        before = found.text == r["expected"]
        after = proposal == r["expected"]
        key = r["truth"]
        s = outcomes[key]
        s["n"] += 1
        s["baseline_correct"] += int(before)
        s["candidate_correct"] += int(after)
        s["helped"] += int(not before and after)
        s["harmed"] += int(before and not after)
        s["changed"] += int(found.text != proposal)
        if found.text != proposal:
            change_records.append({
                "source": r["source"], "sha256": r["sha256"],
                "expected_codec": r["truth"], "original_guess": found.encoding,
                "rerank_action": route, "baseline_correct": before,
                "counterfactual_correct": after,
            })
    totals = {name: sum(v[name] for v in outcomes.values()) for name in (
        "n", "baseline_correct", "candidate_correct", "helped", "harmed", "changed"
    )}
    assert totals["n"] == 102
    report = {
        "schema": "codeccat-c1-original-legacy-v1",
        "status": "previously_inspected_original_files_regression",
        "totals": totals,
        "by_codec": {k: dict(v) for k, v in sorted(outcomes.items())},
        "changed_case_metadata": change_records,
        "excluded_file_count": len(skips),
        "raw_content_exported": False,
        "training_overlap_and_prior_inspection": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("C1_ORIGINAL_LEGACY", json.dumps(report["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
