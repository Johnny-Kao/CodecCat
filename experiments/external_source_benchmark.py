"""External-source benchmark: published Stable 0.1.0, no retraining.

Protocol: research/SOURCE_DISJOINT_BENCHMARK_PROTOCOL_2026-11-07.md
Only metadata and scores are saved; external source bytes remain in Actions.
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
import charset_normalizer
import codeccat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets" / "official"))
from audit_holdout_overlap import blocks

NAUTILUS_COMMIT = "6412aa2752fa91309b4d6e25addb8ed11afd6e69"
CPYTHON_COMMIT = "f64e51563a85667446793ad61ea521d005ceb171"
TRAIN_COMMIT = "b0c0d206913ba118ffea2f7d306883e7a4a76db7"
MANIFEST_HASH = "8ec3007dbd59336cee6018cfa44aeded39cca205e93f5fc9f24ee5ecf87f3ca9"
OUTPUT = Path("research-results/external-source-benchmark.json")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_inputs():
    result = []
    rejected = []
    po_files = sorted(Path("nautilus-confirmation/po").glob("*.po"))
    if len(po_files) != 124:
        raise ValueError(f"pinned Nautilus .po count expected 124; got {len(po_files)}")
    for path in po_files:
        raw = path.read_bytes()
        try:
            truth = raw.decode("utf-8", "strict")
        except UnicodeError:
            rejected.append({"name": path.name, "reason": "not_strict_utf8"})
            continue
        if b"charset=utf-8" not in raw.lower():
            rejected.append({"name": path.name, "reason": "missing_utf8_charset"})
            continue
        result.append(("nautilus_po_utf8", path.name, "utf-8", raw, truth))
    if len(result) < 100:
        raise ValueError("unexpectedly few structurally valid Nautilus PO files")
    references = sorted(Path("cpython-confirmation/Lib/test/cjkencodings").glob("*-utf8.txt"))
    if len(references) != 15:
        raise ValueError(f"pinned CPython pair count expected 15; got {len(references)}")
    for path in references:
        name = path.name.removesuffix("-utf8.txt")
        source = path.with_name(name + ".txt")
        codec = codecs.lookup(name).name
        legacy = source.read_bytes()
        utf8 = path.read_bytes()
        truth = utf8.decode("utf-8", "strict")
        if legacy.decode(codec, "strict") != truth:
            raise ValueError(f"CPython source/reference mismatch: {name}")
        result.append(("cpython_cjk_legacy", name, codec, legacy, truth))
        result.append(("cpython_cjk_utf8", path.name, "utf-8", utf8, truth))
    return result, rejected


def screen(inputs):
    manifest_path = Path("source-data/codeccat-replay-manifest.jsonl")
    if digest(manifest_path.read_bytes()) != MANIFEST_HASH:
        raise ValueError("training manifest changed")
    records = [json.loads(s) for s in manifest_path.read_text().splitlines() if s.strip()]
    if len(records) != 3086:
        raise ValueError("training-record count mismatch")
    root = manifest_path.parent.resolve()
    full, prefix, spans = set(), set(), defaultdict(set)
    for row in records:
        rel = Path(row["path"])
        file = (root / rel).resolve()
        if rel.is_absolute() or not file.is_relative_to(root):
            raise ValueError("unsafe training path")
        raw = file.read_bytes()
        if digest(raw) != row["sha256"] or digest(raw[:4096]) != row["prefix_sha256"]:
            raise ValueError(f"training source changed: {rel}")
        full.add(row["sha256"])
        prefix.add(row["prefix_sha256"])
        for block in blocks(raw[:4096]):
            spans[block].add(row["path"])
    audit = []
    for pool, name, codec, raw, _ in inputs:
        overlaps = Counter(f for block in blocks(raw) for f in spans.get(block, ()))
        audit.append({
            "source": pool, "name": name, "truth_codec": codec, "size": len(raw),
            "sha256": digest(raw),
            "exact_full": digest(raw) in full,
            "exact_prefix": digest(raw[:4096]) in prefix,
            "sampled_near_candidate": max(overlaps.values(), default=0) >= 3,
        })
    bad = [x for x in audit if x["exact_full"] or x["exact_prefix"] or x["sampled_near_candidate"]]
    if bad:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps({"status": "overlap_detected_no_scoring",
                                      "bad": bad, "n": len(audit)}, indent=2) + "\n")
        raise ValueError(f"overlap detected in {len(bad)} external samples; stopped before scoring")
    return audit


def competitor_text(raw, engine):
    try:
        if engine == "chardet":
            encoding = chardet.detect(raw).get("encoding")
        else:
            best = charset_normalizer.from_bytes(raw).best()
            encoding = best.encoding if best is not None else None
        return raw.decode(encoding, "strict").lstrip("\ufeff") if encoding else None, encoding
    except (UnicodeError, LookupError, ValueError, TypeError):
        return None, None


def main():
    versions = {
        "codeccat": "0.1.0", "chardet": "7.7.0",
        "charset-normalizer": "3.5.2",
    }
    for name, expected in versions.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"{name}: unexpected version {actual}; expected {expected}")
    inputs, rejected = load_inputs()
    audit = screen(inputs)  # No predictions before the complete overlap screen.
    group_counts = defaultdict(lambda: defaultdict(int))
    outcomes = []
    detector = codeccat.CodecCat()
    for (pool, name, codec, raw, truth), audit_row in zip(inputs, audit, strict=True):
        ours = detector.detect(raw)
        ours_ok = ours.text == truth
        ch_text, ch_label = competitor_text(raw, "chardet")
        cn_text, cn_label = competitor_text(raw, "charset_normalizer")
        results = {"codeccat": ours_ok, "chardet": ch_text == truth,
                   "charset_normalizer": cn_text == truth}
        group_counts[pool]["n"] += 1
        for engine, ok in results.items():
            group_counts[pool][engine + "_correct"] += int(ok)
        outcomes.append({
            **audit_row, "correct": results,
            "codeccat_guess": ours.encoding, "chardet_guess": ch_label,
            "charset_normalizer_guess": cn_label,
        })
    report = {
        "schema": "codeccat-external-source-benchmark-v1",
        "status": "screen_negative_evaluated",
        "published_package_versions": versions,
        "source_commits": {"nautilus": NAUTILUS_COMMIT, "cpython": CPYTHON_COMMIT,
                           "historical_training": TRAIN_COMMIT},
        "eligible": len(inputs), "structurally_rejected": rejected,
        "groups": {k: dict(v) for k, v in sorted(group_counts.items())},
        "inputs_metadata_only": outcomes, "raw_content_exported": False,
        "limitations": "No semantic independence proof; historical CPython previously evaluated; Nautilus all UTF-8 and non-representative; original 843 remains development-only.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("EXTERNAL_SOURCE_BENCHMARK", json.dumps({
        "status": report["status"], "eligible": report["eligible"],
        "rejected": len(rejected), "groups": report["groups"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
