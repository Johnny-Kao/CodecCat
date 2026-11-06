"""Research-only per-case metadata for published Stable text mismatches.

Pinned development sources are read within CI only. No original bytes, Unicode
text, or retrained model is written to artifacts.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import chardet
import codeccat

import gtk_blind_holdout as gtk
import uchardet_blind_holdout as uch


OUT = Path("research-results/stable-failure-diagnostics.json")


def mismatch_kind(actual: str | None, expected: str) -> str:
    if actual is None:
        return "no_decoded_text"
    if actual == expected:
        return "correct"
    if actual.lstrip("\ufeff") == expected:
        return "bom_only"
    if actual.replace("\r\n", "\n") == expected.replace("\r\n", "\n"):
        return "newline_only"
    return "different_text"


def analyze(source: str, entries: list[dict]) -> list[dict]:
    detector = codeccat.CodecCat()
    failures = []
    for entry in entries:
        raw, expected, truth = entry["raw"], entry["expected"], entry["truth"]
        result = detector.detect(raw)
        if result.text == expected:
            continue
        guess = gtk.norm(result.encoding)
        # Decode using only the Runtime's predicted codec, with the same strict
        # BOM handling as the existing historical evaluation.
        expected_under_guess = gtk.decoded(raw, guess)
        competitor = gtk.norm(chardet.detect(raw).get("encoding"))
        competitor_text = gtk.decoded(raw, competitor)
        failures.append({
            "source": source,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_size": len(raw),
            "truth_encoding": truth,
            "runtime_guess": guess,
            "chardet_guess": competitor,
            "chardet_correct_text": competitor_text == expected,
            "runtime_text_matches_own_codec_decode": result.text == expected_under_guess,
            "runtime_text_mismatch_kind": mismatch_kind(result.text, expected),
            "own_codec_decode_mismatch_kind": mismatch_kind(expected_under_guess, expected),
            "correct_under_truth_codec": gtk.decoded(raw, truth) == expected,
        })
    return failures


def main() -> None:
    if importlib.metadata.version("codeccat") != "0.1.0":
        raise ValueError("requires released codeccat==0.1.0")
    gtk_entries = [
        {"raw": row["bytes"], "expected": row["truth_text"], "truth": row["truth_label"]}
        for row in gtk.cases()
    ]
    uch_rows, _ = uch.source_samples()
    uch_entries = [
        {"raw": row["raw"], "expected": row["expected"], "truth": row["truth"]}
        for row in uch_rows
    ]
    if len(gtk_entries) != 328 or len(uch_entries) != 102:
        raise ValueError("pinned development source selection changed")
    gtk_failures = analyze("gtk", gtk_entries)
    uch_failures = analyze("uchardet", uch_entries)
    if len(gtk_failures) != 4 or len(uch_failures) != 7:
        raise ValueError("published Stable failures changed; inspect before attributing")
    report = {
        "schema": "codeccat-stable-failure-diagnostics-v1",
        "runtime_version": "0.1.0",
        "previously_observed_data": True,
        "stores_raw_content": False,
        "expected_sample_counts": {"gtk": 328, "uchardet": 102},
        "failures": gtk_failures + uch_failures,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("FAILURE_DIAGNOSTICS", json.dumps({
        "gtk": len(gtk_failures), "uchardet": len(uch_failures),
        "own_codec_decode_mismatch": sum(
            not row["runtime_text_matches_own_codec_decode"] for row in report["failures"]
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
