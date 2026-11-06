"""Pinned CPython source confirmation; preregistered before scoring.

Only metadata may be exported. Any historical raw-overlap indicator makes
this source ineligible for independent confirmation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import codecs
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import chardet
import codeccat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets" / "official"))
from audit_holdout_overlap import blocks

ROOT = Path("cpython-confirmation/Lib/test/cjkencodings")
TRAIN = Path("source-data/codeccat-replay-manifest.jsonl")
OUT = Path("research-results/cpython-encoding-confirmation.json")
COMMIT = "f64e51563a85667446793ad61ea521d005ceb171"
MANIFEST_HASH = "8ec3007dbd59336cee6018cfa44aeded39cca205e93f5fc9f24ee5ecf87f3ca9"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inputs() -> list[tuple[str, str, bytes, str]]:
    references = sorted(ROOT.glob("*-utf8.txt"))
    if len(references) != 15:
        raise ValueError(f"expected 15 pinned paired references, found {len(references)}")
    output = []
    for ref in references:
        name = ref.name.removesuffix("-utf8.txt")
        source = ROOT / (name + ".txt")
        if not source.is_file():
            raise ValueError(f"missing reference input: {name}")
        codec = codecs.lookup(name).name
        raw = source.read_bytes()
        utf8_raw = ref.read_bytes()
        if raw.decode(codec, "strict") != utf8_raw.decode("utf-8", "strict"):
            raise ValueError(f"reference decoded-text mismatch: {name}")
        output.append((name, codec, raw, utf8_raw.decode("utf-8")))
        output.append((name + "-utf8", "utf-8", utf8_raw, utf8_raw.decode("utf-8")))
    return output


def overlap_check(rows: list[tuple[str, str, bytes, str]]) -> list[dict]:
    if digest(TRAIN.read_bytes()) != MANIFEST_HASH:
        raise ValueError("pinned training manifest changed")
    manifest = [json.loads(s) for s in TRAIN.read_text().splitlines() if s.strip()]
    if len(manifest) != 3086:
        raise ValueError("training count differs from frozen historical manifest")
    full, prefix, spans = set(), set(), defaultdict(set)
    root = TRAIN.parent.resolve()
    for record in manifest:
        rel = Path(record["path"])
        path = (root / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(root):
            raise ValueError("unsafe training path")
        raw = path.read_bytes()
        if digest(raw) != record["sha256"] or digest(raw[:4096]) != record["prefix_sha256"]:
            raise ValueError("training source hash mismatch")
        full.add(record["sha256"])
        prefix.add(record["prefix_sha256"])
        for block in blocks(raw[:4096]):
            spans[block].add(record["path"])
    results = []
    for name, codec, raw, _ in rows:
        sample_blocks = blocks(raw)
        candidates = Counter(
            file for block in sample_blocks for file in spans.get(block, ())
        )
        result = {
            "name": name,
            "source_sha256": digest(raw),
            "length": len(raw),
            "truth_codec": codec,
            "exact_full": digest(raw) in full,
            "exact_prefix": digest(raw[:4096]) in prefix,
            "sampled_near_candidate": max(candidates.values(), default=0) >= 3,
        }
        results.append(result)
    return results


def utf8_counterfactual(raw: bytes, detection, minimum: int) -> str | None:
    # Same prerequisites as the existing published rule except non-ASCII count.
    if len(raw) < 64 or detection.bom_present or detection.encoding in ("utf-8", "utf-8-sig"):
        return detection.text
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeError:
        return detection.text
    if any((ord(c) < 32 and c not in "\t\n\r") or 0x7f <= ord(c) < 0xa0 for c in text):
        return detection.text
    return text if sum(ord(c) >= 128 for c in text) >= minimum else detection.text


def main() -> None:
    if importlib.metadata.version("codeccat") != "0.1.0":
        raise ValueError("published Stable 0.1.0 required")
    rows = inputs()
    checked = overlap_check(rows)
    report = {
        "schema": "codeccat-preregistered-cpython-confirmation-v1",
        "source_revision": COMMIT,
        "source_dir": "Lib/test/cjkencodings",
        "source_entries": checked,
        "overlap_screen": "exact full, first 4096 bytes, >=3 shared 128-byte 64-step blocks",
        "limitations": "No semantic independence guarantee; excludes Latin single-byte texts",
        "raw_content_exported": False,
        "scores": None,
        "status": "overlap_screen_pending",
    }
    contaminated = [
        r["name"] for r in checked
        if r["exact_full"] or r["exact_prefix"] or r["sampled_near_candidate"]
    ]
    if contaminated:
        report["status"] = "not_independent_overlap_candidates"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise ValueError(f"CPython source not clean under preregistered overlap screen: {len(contaminated)} cases")
    detector = codeccat.CodecCat()
    scores = {name: Counter() for name in ("baseline", "chardet", "threshold_1", "threshold_2", "threshold_3")}
    by_codec = {}
    failure_diagnostics = []
    trained_labels = set(str(c) for c in detector._model.classifier.classes_)
    for name, codec, raw, truth in rows:
        found = detector.detect(raw)
        baseline = found.text == truth
        chardet_guess = chardet.detect(raw).get("encoding")
        try:
            competitor = raw.decode(chardet_guess, "strict").lstrip("\ufeff") == truth
        except (UnicodeError, LookupError, TypeError):
            competitor = False
        if not baseline:
            failure_diagnostics.append({
                "name": name,
                "truth_codec": codec,
                "truth_codec_in_trained_classes": codec in trained_labels,
                "runtime_guess": found.encoding,
                "chardet_guess": chardet_guess,
                "runtime_abstained": found.text is None,
            })
        options = {
            "baseline": baseline,
            "chardet": competitor,
            **{
                f"threshold_{t}": utf8_counterfactual(raw, found, t) == truth
                for t in (1, 2, 3)
            },
        }
        metrics = by_codec.setdefault(codec, Counter())
        metrics["n"] += 1
        for label, correct in options.items():
            scores[label]["correct"] += int(correct)
            metrics[label + "_correct"] += int(correct)
            if label.startswith("threshold_"):
                scores[label]["helped"] += int(correct and not baseline)
                scores[label]["harmed"] += int(baseline and not correct)
    report["scores"] = {k: dict(v) for k, v in scores.items()}
    report["by_codec"] = {k: dict(v) for k, v in by_codec.items()}
    report["failures"] = failure_diagnostics
    report["status"] = "screen_negative_evaluated"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CPYTHON_CONFIRMATION", json.dumps({
        "cases": len(rows), "status": report["status"], "scores": report["scores"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
