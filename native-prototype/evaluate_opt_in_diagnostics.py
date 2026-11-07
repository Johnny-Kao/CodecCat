"""Opt-in warning evaluation on original labeled encoding files from other projects.

The original files are fixture files, NOT independently sampled production
documents. They remain unchanged and are downloaded only in GitHub Actions.
This script does not modify the model or published default behavior.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import codecs
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

import codeccat
from codeccat.diagnostics import inspect_ambiguity

OUT = Path("research-results/staging-c1-labeled-fixtures.json")
SOURCES = (
    ("zed-encodings-tests", Path("fixture-zed")),
    ("char-test", Path("fixture-char-test/test_files")),
)
SAMPLES = {
    "iso88591.txt": "iso8859-1",
    "windows1252.txt": "cp1252",
}


def normalize(name):
    try:
        return codecs.lookup(name).name
    except (LookupError, TypeError):
        return None


def get_inputs():
    rows = []
    skipped = []
    for name, root in SOURCES:
        if name == "zed-encodings-tests":
            files = sorted(root.glob("*.txt"))
            if len(files) < 25:
                raise ValueError("unexpectedly few pinned fixture files")
            for path in files:
                codec = normalize(path.stem.replace("ISO_", "ISO-").replace("_", "-"))
                if codec is None:
                    skipped.append({"source": name, "path": path.name, "reason": "unsupported_python_codec"})
                    continue
                raw = path.read_bytes()
                try:
                    truth = raw.decode(codec, "strict").lstrip("\ufeff")
                except UnicodeError:
                    skipped.append({"source": name, "path": path.name, "reason": "declared_encoding_invalid"})
                    continue
                rows.append((name, path.name, codec, raw, truth))
        else:
            for filename, label in SAMPLES.items():
                path = root / filename
                if not path.is_file():
                    raise ValueError(f"required fixture absent: {filename}")
                raw = path.read_bytes()
                rows.append((name, filename, normalize(label), raw, raw.decode(label, "strict")))
    return rows, skipped


def main():
    assert importlib.metadata.version("codeccat") == "0.1.0"
    detector = codeccat.CodecCat()
    inputs, skips = get_inputs()
    results = defaultdict(Counter)
    timing = defaultdict(list)
    details = []
    for origin, filename, codec, raw, expected in inputs:
        result = detector.detect(raw)
        before_text = result.text
        before_codec = result.encoding
        before_candidates = result.candidates
        diag = inspect_ambiguity(raw, result)
        assert result.text == before_text and result.encoding == before_codec
        assert result.candidates == before_candidates
        assert diag.warning == bool(diag.alternative_encodings)
        correct = result.text == expected
        s = results[origin]
        s["n"] += 1
        s["correct"] += int(correct)
        s["incorrect"] += int(not correct)
        s["flagged"] += int(diag.warning)
        s["flagged_correct"] += int(diag.warning and correct)
        s["flagged_incorrect"] += int(diag.warning and not correct)
        # Measurement of incremental warning call, not inference.
        for _ in range(30):
            tic = time.perf_counter_ns()
            inspect_ambiguity(raw, result)
            timing[origin].append((time.perf_counter_ns() - tic) / 1000)
        details.append({
            "source": origin, "name": filename, "expected_encoding": codec,
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "baseline_correct": correct, "flagged": diag.warning,
            "reason": diag.reason, "alternatives": list(diag.alternative_encodings),
        })
    if len(inputs) < 25 or not results["char-test"]["n"] == 2:
        raise ValueError("unexpected fixture selection")
    report = {
        "schema": "codeccat-opt-in-c1-original-labeled-fixtures-v1",
        "release_version": "0.1.0", "status": "diagnostic_only",
        "groups": {name: dict(s) for name,s in sorted(results.items())},
        "timing_microseconds": {name: {
            "median": round(statistics.median(samples), 3),
            "p95": round(sorted(samples)[int(0.95 * (len(samples)-1))], 3),
            "n_calls": len(samples),
        } for name,samples in timing.items()},
        "skips": skips, "case_metadata": details,
        "raw_content_exported": False,
        "limitations": "Original labeled project fixtures, not independently sampled production files. Small correlated test data, preliminary timing only. Does not establish true-world false-positive rates.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("OPT_IN_C1_LABELED_FIXTURE_RESULTS", json.dumps({
        "groups": report["groups"], "timing": report["timing_microseconds"],
        "skipped": len(skips),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
