"""Research-only counterfactual for UTF-8 rescue; no model or Runtime edits.

Existing GTK/uchardet sets are previously inspected development data, NOT
independent confirmation. Only aggregate paired outcomes and safe flags leave CI.
"""
from __future__ import annotations

from collections import Counter
import importlib.metadata
import json
from pathlib import Path

import codeccat
from codeccat.vnext_decoding import _strong_utf8_evidence

import gtk_blind_holdout as gtk
import uchardet_blind_holdout as uch


OUT = Path("research-results/stable-utf8-counterfactual.json")
THRESHOLDS = (1, 2, 3, 5, 10)


def evidence(raw: bytes) -> dict:
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeError:
        return {"valid": False, "nonascii": 0, "controls": False}
    return {
        "valid": True,
        "nonascii": sum(ord(c) >= 128 for c in text),
        "controls": any(
            (ord(c) < 32 and c not in "\t\n\r") or (0x7F <= ord(c) < 0xA0)
            for c in text
        ),
    }


def collect() -> dict:
    detector = codeccat.CodecCat()
    gtk_rows = [
        (r["bytes"], r["truth_text"]) for r in gtk.cases()
    ]
    uch_rows, _ = uch.source_samples()
    sets = {
        "gtk": gtk_rows,
        "uchardet": [(r["raw"], r["expected"]) for r in uch_rows],
    }
    if (len(gtk_rows), len(uch_rows)) != (328, 102):
        raise ValueError("source selection changed")
    output = {}
    for source, rows in sets.items():
        per_threshold = {
            str(k): Counter() for k in THRESHOLDS
        }
        failures = []
        baseline_correct = 0
        for raw, truth in rows:
            result = detector.detect(raw)
            correct = result.text == truth
            baseline_correct += correct
            ev = evidence(raw)
            existing_rule = _strong_utf8_evidence(raw)
            candidate = (
                len(raw) >= 64
                and not result.bom_present
                and result.encoding not in ("utf-8", "utf-8-sig")
                and ev["valid"]
                and not ev["controls"]
            )
            if not correct:
                failures.append({
                    "predicted_encoding": result.encoding,
                    "utf8_valid": ev["valid"],
                    "utf8_nonascii_count": ev["nonascii"],
                    "utf8_control_present": ev["controls"],
                    "existing_rescue_evidence": existing_rule,
                    "possible_utf8_candidate": candidate,
                    "utf8_would_match_truth": (
                        raw.decode("utf-8") == truth if ev["valid"] else False
                    ),
                })
            for threshold in THRESHOLDS:
                stats = per_threshold[str(threshold)]
                use_utf8 = candidate and ev["nonascii"] >= threshold
                if use_utf8:
                    stats["selected"] += 1
                revised = raw.decode("utf-8") if use_utf8 else result.text
                revised_correct = revised == truth
                stats["helped"] += int(revised_correct and not correct)
                stats["harmed"] += int(correct and not revised_correct)
                stats["correct"] += int(revised_correct)
        output[source] = {
            "n": len(rows),
            "baseline_correct": baseline_correct,
            "counterfactual": {k: dict(v) for k, v in per_threshold.items()},
            "failure_signals": failures,
        }
    return output


def main() -> None:
    if importlib.metadata.version("codeccat") != "0.1.0":
        raise ValueError("published stable 0.1.0 required")
    r = {
        "schema": "stable-utf8-rescue-counterfactual-v1",
        "development_only": True,
        "raw_content_exported": False,
        "data": collect(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("UTF8_COUNTERFACTUAL", json.dumps({
        source: {"baseline": data["baseline_correct"],
                 "thresholds": data["counterfactual"]}
        for source, data in r["data"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
