"""Inspect Western mistakes on already scored samples; no decoder changes.

Runs after the existing pinned overall prepare phase. Metadata only.
All results are engineering diagnostics, not an independent score.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path

import chardet
import codeccat
from charset_normalizer import from_bytes

WORK = Path(".overall-work")
OUT = Path("research-results/western-legacy-diagnosis.json")


def norm(codec):
    import codecs
    try:
        return codecs.lookup(codec).name if codec else None
    except (LookupError, TypeError):
        return None


def correct_text(raw, codec):
    try:
        return raw.decode(codec, "strict") if codec else None
    except (LookupError, UnicodeError):
        return None


def main():
    assert importlib.metadata.version("codeccat") == "0.1.0"
    assert importlib.metadata.version("chardet") == "7.7.0"
    manifest = json.loads((WORK / "manifest.json").read_text())
    runtime = codeccat.CodecCat()
    summary = defaultdict(Counter)
    wrong_guesses = defaultdict(Counter)
    disagreements = defaultdict(Counter)
    by_locale = defaultdict(Counter)
    utf8_properties = defaultdict(Counter)
    cases = []
    for row in manifest:
        raw = (WORK / row["file"]).read_bytes()
        truth = correct_text(raw, row["encoding"])
        # Predeclared independent expected text fingerprint.
        assert hashlib.sha256(truth.encode("utf-8")).hexdigest() == row["truth_sha256"]
        got = runtime.detect(raw)
        label = norm(got.encoding)
        is_correct = got.text == truth
        ref_guess = norm(chardet.detect(raw).get("encoding"))
        ref_text = correct_text(raw, ref_guess)
        if ref_text is not None:
            ref_text = ref_text.lstrip("\ufeff")
        ref_correct = ref_text == truth
        utf8 = correct_text(raw, "utf-8")
        utf8_nonascii = sum(ord(c) >= 128 for c in utf8) if utf8 else 0
        utf8_valid = utf8 is not None
        is_west = row["pool"] == "western_synthetic"
        if is_west:
            codec = row["encoding"]
            locale = row["source"].split(".po#")[0]
            summary[codec]["n"] += 1
            summary[codec]["codeccat_correct"] += int(is_correct)
            summary[codec]["chardet_correct"] += int(ref_correct)
            wrong_guesses[codec][label or "abstain"] += int(not is_correct)
            disagreements[codec]["chardet_correct_codecCat_wrong"] += int(ref_correct and not is_correct)
            disagreements[codec]["chardet_wrong_codecCat_correct"] += int(not ref_correct and is_correct)
            by_locale[locale]["n"] += 1
            by_locale[locale]["codeccat_wrong"] += int(not is_correct)
            by_locale[locale]["chardet_wrong"] += int(not ref_correct)
            utf8_properties[codec]["utf8_valid"] += int(utf8_valid)
            utf8_properties[codec]["utf8_valid_and_wrong"] += int(utf8_valid and not is_correct)
            utf8_properties[codec]["utf8_3plus_and_wrong"] += int(
                utf8_valid and utf8_nonascii >= 3 and not is_correct
            )
            # Exact same bytes; counterfactual rule: treat *only* strict UTF8 as override
            # (even if it would not be wise), to determine whether UTF8 ambiguity caused errors.
            utf8_properties[codec]["utf8_override_would_help"] += int(
                not is_correct and utf8 == truth
            )
            utf8_properties[codec]["utf8_override_would_harm"] += int(
                is_correct and utf8_valid and utf8 != truth
            )
            if not is_correct:
                cases.append({
                    "source": row["source"], "expected_codec": codec,
                    "runtime_guess": label, "competitor_guess": ref_guess,
                    "competitor_correct": ref_correct, "utf8_valid": utf8_valid,
                    "utf8_nonascii_count": utf8_nonascii,
                    "length": row["size"], "sha256": row["sha256"],
                    "ambiguous": bool(got.ambiguous),
                    "reason": str(got.reason),
                    "runtime_alternatives": list(got.candidates)[:5],
                    "raw_exported": False,
                })
    report = {
        "schema": "codeccat-western-legacy-diagnosis-v1",
        "source": "same preregistered overall benchmark, previously inspected",
        "status": "diagnostic_only",
        "by_expected_codec": {k: dict(v) for k,v in sorted(summary.items())},
        "wrong_guess_distribution": {k: dict(v) for k,v in sorted(wrong_guesses.items())},
        "chardet_disagreements": {k: dict(v) for k,v in sorted(disagreements.items())},
        "by_source_locale": {k: dict(v) for k,v in sorted(by_locale.items())},
        "utf8_properties": {k: dict(v) for k,v in sorted(utf8_properties.items())},
        "wrong_case_metadata": cases,
        "raw_content_exported": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("WESTERN_DIAGNOSIS", json.dumps({
        "by_codec": report["by_expected_codec"],
        "wrong_guesses": report["wrong_guess_distribution"],
        "chardet_disagreements": report["chardet_disagreements"],
        "utf8": report["utf8_properties"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
