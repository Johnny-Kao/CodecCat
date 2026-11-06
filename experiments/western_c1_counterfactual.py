"""Controlled, research-only paired test of C1-control mitigation.

Only modifies output in memory after the published Stable detector returns.
No runtime modification, no trained-model access, no threshold fitted to inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import codecs
import hashlib
import importlib.metadata
import json
from pathlib import Path

import codeccat

WORK = Path(".overall-work")
OUT = Path("research-results/western-c1-counterfactual.json")


def text_for(raw, encoding):
    try:
        return raw.decode(encoding, "strict").lstrip("\ufeff")
    except (LookupError, UnicodeError, TypeError):
        return None


def c1_count(text):
    return sum(0x80 <= ord(char) <= 0x9f for char in text or "")


def rerank_control(raw, baseline):
    if baseline.text is None or c1_count(baseline.text) == 0:
        return baseline.text, "unchanged"
    # The existing detector already generated alternatives based on the model.
    # Require a strict successful decode free of U+0080..U+009F.
    for candidate in baseline.candidates:
        if str(candidate) == str(baseline.encoding):
            continue
        possible = text_for(raw, candidate)
        if possible is not None and c1_count(possible) == 0:
            return possible, "reranked"
    return baseline.text, "no_clean_candidate"


def main():
    assert importlib.metadata.version("codeccat") == "0.1.0"
    det = codeccat.CodecCat()
    rows = json.loads((WORK / "manifest.json").read_text())
    per_pool = defaultdict(lambda: defaultdict(Counter))
    by_codec = defaultdict(lambda: defaultdict(Counter))
    by_route = defaultdict(Counter)
    changes = []
    for row in rows:
        raw = (WORK / row["file"]).read_bytes()
        if row["pool"] == "natural_utf8":
            expected = raw.decode("utf-8", "strict")
        elif row["pool"] == "cjk_natural_preexamined":
            name = row["source"].removesuffix(".txt")
            expected = (Path("external-cpython/Lib/test/cjkencodings") /
                        (name + "-utf8.txt")).read_text(encoding="utf-8")
        else:
            expected = raw.decode(row["encoding"], "strict")
        assert hashlib.sha256(expected.encode("utf-8")).hexdigest() == row["truth_sha256"]
        baseline = det.detect(raw)
        baseline_ok = baseline.text == expected
        reranked, route = rerank_control(raw, baseline)
        rerank_ok = reranked == expected
        # Abstain instead of returning suspect C1 characters (only if present).
        abstain = None if c1_count(baseline.text) else baseline.text
        abstain_ok = abstain == expected
        stats = per_pool[row["pool"]]
        codec_stats = by_codec[row["encoding"]]
        for target in (stats, codec_stats):
            target["baseline"]["correct"] += int(baseline_ok)
            target["baseline"]["wrong_nonempty"] += int(baseline.text is not None and not baseline_ok)
            target["rerank"]["correct"] += int(rerank_ok)
            target["rerank"]["helped"] += int(rerank_ok and not baseline_ok)
            target["rerank"]["harmed"] += int(baseline_ok and not rerank_ok)
            target["rerank"]["wrong_nonempty"] += int(reranked is not None and not rerank_ok)
            target["abstain"]["correct"] += int(abstain_ok)
            target["abstain"]["helped"] += int(abstain_ok and not baseline_ok)
            target["abstain"]["harmed"] += int(baseline_ok and not abstain_ok)
            target["abstain"]["abstentions"] += int(abstain is None)
            target["abstain"]["wrong_nonempty"] += int(abstain is not None and not abstain_ok)
            target["total"]["n"] += 1
        by_route[row["pool"]][route] += 1
        if reranked != baseline.text or abstain != baseline.text:
            changes.append({
                "source": row["source"], "pool": row["pool"],
                "expected_codec": row["encoding"], "sha256": row["sha256"],
                "chosen_codec": baseline.encoding, "c1_chars": c1_count(baseline.text),
                "rerank_action": route, "baseline_correct": baseline_ok,
                "rerank_correct": rerank_ok, "abstain_correct": abstain_ok,
            })
    report = {
        "schema": "codeccat-c1-control-counterfactual-v1",
        "status": "research_only_no_runtime_change",
        "original_published_version": "0.1.0",
        "pools": {p: {variant: dict(counts) for variant, counts in variants.items()}
                  for p, variants in sorted(per_pool.items())},
        "by_codec": {p: {variant: dict(counts) for variant, counts in variants.items()}
                     for p, variants in sorted(by_codec.items())},
        "routes": {p: dict(x) for p,x in sorted(by_route.items())},
        "changed_case_metadata": changes,
        "raw_content_exported": False,
        "limitations": "Previously examined data; no generalization claim. Strict C1 control guard is a counterfactual, not a recommendation to change contract.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("WESTERN_C1_COUNTERFACTUAL", json.dumps({
        "pools": report["pools"], "routes": report["routes"],
        "cp1252": report["by_codec"].get("cp1252"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
