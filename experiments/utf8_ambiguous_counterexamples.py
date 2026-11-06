"""Synthetic counterexamples to UTF-8 byte-validity based auto-correction.

These are explicitly constructed CP1252 strings whose encoded bytes also form
valid UTF-8; their intended text is known from how they were constructed.
They are NOT representative population estimates or blind test examples.
"""
from __future__ import annotations

from collections import Counter
import importlib.metadata
import json
from pathlib import Path

import codeccat

OUT = Path("research-results/ambiguous-cp1252-counterexamples.json")
MARKERS = ("Â©", "Â®", "Â£", "Ã©", "Ã±")
PREFIXES = (
    "The document contains the characters ",
    "This legacy archive deliberately retains ",
    "A plain English paragraph records ",
    "The report preserves the literal sequence ",
)


def attempt_utf8(raw: bytes, result, threshold: int) -> str | None:
    if len(raw) < 64 or result.bom_present or result.encoding in ("utf-8", "utf-8-sig"):
        return result.text
    try:
        decoded = raw.decode("utf-8", "strict")
    except UnicodeError:
        return result.text
    if any((ord(ch) < 32 and ch not in "\t\n\r") or 0x7f <= ord(ch) < 0xa0 for ch in decoded):
        return result.text
    return decoded if sum(ord(ch) >= 128 for ch in decoded) >= threshold else result.text


def main() -> None:
    if importlib.metadata.version("codeccat") != "0.1.0":
        raise ValueError("must run against published Runtime")
    det = codeccat.CodecCat()
    summary = {str(t): Counter() for t in (1, 2, 3)}
    n = baseline_correct = 0
    for prefix in PREFIXES:
        for marker in MARKERS:
            for count in (1, 2, 3):
                intended = (prefix + " / ".join([marker] * count) +
                            ". This paragraph was written using Windows-1252. " * 2)
                raw = intended.encode("cp1252")
                assert raw.decode("utf-8", "strict") != intended
                result = det.detect(raw)
                correct = result.text == intended
                n += 1
                baseline_correct += int(correct)
                for threshold in (1, 2, 3):
                    cf = attempt_utf8(raw, result, threshold)
                    revised = cf == intended
                    stats = summary[str(threshold)]
                    stats["correct"] += int(revised)
                    stats["harmed"] += int(correct and not revised)
                    stats["helped"] += int(revised and not correct)
                    stats["changed"] += int(cf != result.text)
    report = {
        "schema": "codeccat-ambiguous-cp1252-synthetic-v1",
        "synthetic": True,
        "representative": False,
        "raw_content_exported": False,
        "cases": n,
        "baseline_correct": baseline_correct,
        "thresholds": {k: dict(v) for k, v in summary.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("AMBIGUOUS_CP1252", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
