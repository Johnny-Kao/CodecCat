"""Research-only warning check on historical, genuinely web-crawled HTML.

The archive's original source labels are proxies for expected Unicode text,
NOT verified textual ground truth. No raw HTML or decoded text is exported.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import codecs
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import statistics
import sys
import time
import zipfile

import codeccat
from codeccat.diagnostics import inspect_ambiguity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets" / "official"))
from audit_holdout_overlap import blocks

ARCHIVE = Path("external-iust/eval/test-data/encoding-wise/corpus.zip")
TRAIN = Path("source-data/codeccat-replay-manifest.jsonl")
OUT = Path("research-results/staging-c1-real-html.json")
TRAIN_HASH = "8ec3007dbd59336cee6018cfa44aeded39cca205e93f5fc9f24ee5ecf87f3ca9"
ARCHIVE_GROUPS = {
    "GBK": "gbk",
    "Shift_JIS": "shift_jis",
    "Windows-1251": "cp1251",
    "Windows-1256": "cp1256",
    "UTF-8": "utf-8",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def train_hashes():
    if digest(TRAIN.read_bytes()) != TRAIN_HASH:
        raise ValueError("training manifest mismatch")
    rows = [json.loads(line) for line in TRAIN.read_text().splitlines() if line.strip()]
    if len(rows) != 3086:
        raise ValueError("unexpected historical training input count")
    root = TRAIN.parent.resolve()
    full, prefixes, spans = set(), set(), defaultdict(set)
    for item in rows:
        rel = Path(item["path"])
        file = (root / rel).resolve()
        if rel.is_absolute() or not file.is_relative_to(root):
            raise ValueError("path escapes historical corpus")
        raw = file.read_bytes()
        if digest(raw) != item["sha256"] or digest(raw[:4096]) != item["prefix_sha256"]:
            raise ValueError("historical training source changed")
        full.add(item["sha256"])
        prefixes.add(item["prefix_sha256"])
        for b in blocks(raw[:4096]):
            spans[b].add(item["path"])
    return full, prefixes, spans


def truth_from_label(raw: bytes, codec: str) -> str:
    # UTF-8 BOM is a transport signature, not a character for the default
    # bom_policy='strip' API.
    return raw.decode("utf-8-sig" if codec == "utf-8" else codec, "strict")


def prepare(z, train):
    full, prefixes, spans = train
    info = z.infolist()
    if len(info) > 10000 or sum(x.file_size for x in info) > 500_000_000:
        raise ValueError("archive outside preregistered safety limits")
    counts = defaultdict(Counter)
    seen = set()
    eligible = []
    for member in sorted(info, key=lambda x: x.filename):
        path = PurePosixPath(member.filename)
        parts = path.parts
        group = parts[1] if len(parts) == 3 else None
        if member.is_dir() or group not in ARCHIVE_GROUPS or parts[0] != "corpus" or ".." in parts:
            counts["not_in_scope"]["skipped_structure"] += 1
            continue
        codec = ARCHIVE_GROUPS[group]
        s = counts[group]
        s["all_members"] += 1
        if member.file_size == 0 or member.file_size > 6_000_000:
            s["excluded_size"] += 1
            continue
        raw = z.read(member)
        fp = digest(raw)
        if fp in seen:
            s["excluded_duplicate_bytes"] += 1
            continue
        seen.add(fp)
        try:
            truth_from_label(raw, codec)
        except UnicodeError:
            s["excluded_invalid_labeled_decode"] += 1
            continue
        near = Counter(p for block in blocks(raw) for p in spans.get(block, ()))
        if fp in full or digest(raw[:4096]) in prefixes or max(near.values(), default=0) >= 3:
            s["excluded_training_overlap_or_candidate"] += 1
            continue
        s["eligible"] += 1
        eligible.append((member.filename, group, codec, fp, member.file_size))
    return eligible, counts


def evaluate(z, eligible, counts):
    detector = codeccat.CodecCat()
    timing = defaultdict(list)
    meta = []
    for name, group, codec, fp, byte_len in eligible:
        raw = z.read(name)
        if digest(raw) != fp:
            raise ValueError("archive content unexpectedly changed after screen")
        truth = truth_from_label(raw, codec)
        found = detector.detect(raw)
        original = (
            found.encoding, found.text, found.candidates,
            found.ambiguous, found.reason, found.bom_encoding,
            found.bom_present, found.bom_policy, found.l3_called,
            found.l3_changed
        )
        start = time.perf_counter_ns()
        warning = inspect_ambiguity(raw, found)
        elapsed_us = (time.perf_counter_ns() - start) / 1000
        timing[group].append(elapsed_us)
        assert original == (
            found.encoding, found.text, found.candidates,
            found.ambiguous, found.reason, found.bom_encoding,
            found.bom_present, found.bom_policy, found.l3_called,
            found.l3_changed
        )
        agrees = found.text == truth
        s = counts[group]
        s["baseline_label_agreeing_unicode"] += int(agrees)
        s["baseline_label_disagreeing_unicode"] += int(not agrees)
        s["warning"] += int(warning.warning)
        s["warning_label_agreeing"] += int(warning.warning and agrees)
        s["warning_label_disagreeing"] += int(warning.warning and not agrees)
        s["unwarned_label_disagreeing"] += int(not warning.warning and not agrees)
        meta.append({
            "path": name, "codec": codec, "sha256": fp, "bytes": byte_len,
            "label_agreeing": agrees, "warned": warning.warning,
            "reason": warning.reason,
        })
    return timing, meta


def main():
    assert importlib.metadata.version("codeccat") == json.loads(Path("release-config.json").read_text())["version"]
    from importlib.resources import files
    assert digest(files("codeccat").joinpath("models/default.ccm").read_bytes()) == (
        "16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b"
    )
    assert ARCHIVE.stat().st_size == 37123043
    historic = train_hashes()
    with zipfile.ZipFile(ARCHIVE) as z:
        selected, totals = prepare(z, historic)  # No detector call before screen.
        if len(selected) < 200:
            raise ValueError(f"too few strict and overlap-screened source samples: {len(selected)}")
        print("C1_REAL_HTML_ELIGIBLE", json.dumps({
            g: {k:v for k,v in s.items() if k.startswith("excluded_") or k in ("eligible","all_members")}
            for g,s in sorted(totals.items())
        }, sort_keys=True))
        timing, meta = evaluate(z, selected, totals)
    report = {
        "schema": "codeccat-c1-preregistered-historic-real-html-v1",
        "status": "screen_negative_evaluated",
        "source": "IUST-HTMLCharDet/encoding-wise/corpus.zip",
        "source_commit": "0a7f382e2d488d46f1e12bad647f0384c4a2288f",
        "archive_git_blob": "9d6ff0ff9c65c5afc96e1a10b996a51e7c4d3ba5",
        "groups": {k:dict(v) for k,v in sorted(totals.items())},
        "diagnostic_us": {k: {
            "p50": round(statistics.median(v), 2),
            "p95": round(sorted(v)[int(0.95*(len(v)-1))],2),
            "n":len(v),
        } for k,v in sorted(timing.items())},
        "case_metadata": meta,
        "raw_content_exported": False,
        "limitations": "Original web crawl; directory-labeled codecs are not independently verified ground truth. Some HTML can be mislabeled or browser-decodable but not strict Python-decodable. Correlated historical sample.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("C1_REAL_HTML_RESULTS", json.dumps({
        "groups": report["groups"], "diagnostic_us": report["diagnostic_us"],
        "total": len(meta),
    }, sort_keys=True))


if __name__=="__main__":
    main()
