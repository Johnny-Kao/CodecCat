"""Fixed four-cohort published-Wheel benchmark. See research protocol.

Raw test bytes are kept in ephemeral .overall-work/ and never uploaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import codecs
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import statistics
import sys
import time

WORK = Path(".overall-work")
SRC = Path("external-terminal/po")
TRAIN = Path("source-data/codeccat-replay-manifest.jsonl")
CPY = Path("external-cpython/Lib/test/cjkencodings")
RESULTS = Path("research-results/overall-benchmark")
TRAIN_HASH = "8ec3007dbd59336cee6018cfa44aeded39cca205e93f5fc9f24ee5ecf87f3ca9"
MODEL_SHA = "16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b"
WESTERN = {
    **{k: ("cp1252", "iso8859_15") for k in ("fr", "de", "es", "it", "pt", "nl", "sv", "da", "fi")},
    **{k: ("cp1250",) for k in ("pl", "cs", "hu")},
    **{k: ("cp1251",) for k in ("ru", "bg")},
    "el": ("iso8859_7",),
    "tr": ("cp1254",),
}
CJK = {
    "ja": ("shift_jis", "euc_jp", "iso2022_jp"),
    "ko": ("cp949", "euc_kr"),
    "zh_CN": ("gbk", "gb18030"),
    "zh_TW": ("big5", "cp950"),
}
PO_ROOT = "natural_utf8"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canon(value: str | None) -> str | None:
    try:
        return codecs.lookup(value).name if value else None
    except (LookupError, TypeError, ValueError):
        return None


def add(rows, *, pool: str, source: str, codec: str, data: bytes, truth: str, group: str):
    idx = len(rows)
    WORK.mkdir(exist_ok=True)
    path = WORK / f"{idx:05d}.bin"
    path.write_bytes(data)
    rows.append({
        "pool": pool, "source": source, "encoding": canon(codec),
        "sha256": digest(data), "prefix_sha256": digest(data[:4096]),
        "size": len(data), "group": group,
        "file": path.name, "truth_sha256": digest(truth.encode("utf-8")),
    })


def prepare():
    import polib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets" / "official"))
    from audit_holdout_overlap import blocks
    if digest(TRAIN.read_bytes()) != TRAIN_HASH:
        raise ValueError("pinned training manifest mismatch")
    records = [json.loads(s) for s in TRAIN.read_text().splitlines() if s.strip()]
    if len(records) != 3086:
        raise ValueError("expected historical 3086 source records")
    full, prefix, spans = set(), set(), defaultdict(set)
    root = TRAIN.parent.resolve()
    for rec in records:
        rel = Path(rec["path"])
        resolved = (root / rel).resolve()
        if rel.is_absolute() or not resolved.is_relative_to(root):
            raise ValueError("unsafe source path")
        raw = resolved.read_bytes()
        if digest(raw) != rec["sha256"] or digest(raw[:4096]) != rec["prefix_sha256"]:
            raise ValueError("changed historical input")
        full.add(rec["sha256"])
        prefix.add(rec["prefix_sha256"])
        for block in blocks(raw[:4096]):
            spans[block].add(rec["path"])

    po_paths = sorted(SRC.glob("*.po"))
    if len(po_paths) != 100:
        raise ValueError(f"expected 100 pinned PO files, got {len(po_paths)}")
    rows, excludes = [], Counter()
    for path in po_paths:
        raw = path.read_bytes()
        try:
            truth = raw.decode("utf-8", "strict")
        except UnicodeError:
            excludes["non_utf8_natural_po"] += 1
            continue
        if b"charset=utf-8" not in raw.lower():
            excludes["no_charset_utf8"] += 1
            continue
        add(rows, pool=PO_ROOT, source=path.name, codec="utf-8",
            data=raw, truth=truth, group=path.name)

    for pool, mapping in (("western_synthetic", WESTERN), ("cjk_synthetic", CJK)):
        for locale, codecs_to_try in mapping.items():
            path = SRC / (locale + ".po")
            if not path.exists():
                excludes["missing_locale"] += 1
                continue
            po = polib.pofile(str(path), encoding="utf-8", check_for_duplicates=False)
            translated = "\n".join(
                entry.msgstr for entry in po if entry.msgstr and not entry.obsolete
            )
            chunks = [translated[i:i + 1024] for i in range(0, len(translated), 1024)]
            for codec in codecs_to_try:
                accepted = 0
                for j, chunk in enumerate(chunks):
                    if len(chunk) < 128 or sum(ord(c) >= 128 for c in chunk) < 5:
                        excludes["small_or_ascii_synthetic_chunk"] += 1
                        continue
                    try:
                        encoded = chunk.encode(codec, "strict")
                    except UnicodeError:
                        excludes["unencodable_synthetic_chunk"] += 1
                        continue
                    add(rows, pool=pool, source=f"{path.name}#{j}:{codec}", codec=codec,
                        data=encoded, truth=chunk, group=path.name)
                    accepted += 1
                    if accepted >= 8:
                        break
                if not accepted:
                    excludes["zero_sample_locale_codec"] += 1

    reference = sorted(CPY.glob("*-utf8.txt"))
    if len(reference) != 15:
        raise ValueError(f"expected CPython 15 pairs, got {len(reference)}")
    for path in reference:
        name = path.name.removesuffix("-utf8.txt")
        source = path.with_name(name + ".txt")
        raw = source.read_bytes()
        utf8 = path.read_bytes()
        codec = canon(name)
        truth = utf8.decode("utf-8", "strict")
        if not codec or raw.decode(codec, "strict") != truth:
            raise ValueError(f"CPython paired Unicode disagreement: {name}")
        add(rows, pool="cjk_natural_preexamined", source=source.name,
            codec=codec, data=raw, truth=truth, group=source.name)

    audited = []
    for row in rows:
        raw = (WORK / row["file"]).read_bytes()
        neighbors = Counter(path for block in blocks(raw) for path in spans.get(block, ()))
        overlap = {
            "full": row["sha256"] in full,
            "prefix": row["prefix_sha256"] in prefix,
            "sampled_block": max(neighbors.values(), default=0) >= 3,
        }
        audited.append({**row, "overlap": overlap})
    contaminated = [r for r in audited if any(r["overlap"].values())]
    RESULTS.mkdir(parents=True, exist_ok=True)
    if contaminated:
        (RESULTS / "overlap-failure.json").write_text(json.dumps({
            "status": "no_scoring_overlap_detected",
            "count": len(contaminated),
            "flagged_metadata": contaminated,
        }, indent=2))
        raise ValueError(f"historical corpus overlap: {len(contaminated)} flagged; no scoring")

    pools = Counter(r["pool"] for r in rows)
    print("OVERALL_COHORT_PREPARED", json.dumps({
        "total": len(rows), "pools": dict(pools), "group_counts": {
            k: len({r["group"] for r in rows if r["pool"] == k}) for k in pools
        }, "excluded_counts": dict(excludes),
    }, sort_keys=True))
    if any(pools[p] < threshold for p, threshold in (
        ("natural_utf8", 80), ("western_synthetic", 40),
        ("cjk_synthetic", 20), ("cjk_natural_preexamined", 15),
    )):
        raise ValueError(f"insufficient predeclared category coverage: {pools}")
    (WORK / "manifest.json").write_text(json.dumps(audited, indent=2) + "\n")
    (WORK / "selection.json").write_text(json.dumps({
        "rejected": dict(excludes), "pools": dict(pools),
    }, indent=2) + "\n")


def detector_output(engine: str, raw: bytes, instance):
    if engine == "codeccat":
        result = instance.detect(raw)
        return canon(result.encoding), result.text
    if engine == "chardet":
        import chardet
        label = canon(chardet.detect(raw).get("encoding"))
    elif engine == "charset_normalizer":
        from charset_normalizer import from_bytes
        result = from_bytes(raw).best()
        label = canon(result.encoding if result else None)
    else:
        raise ValueError(engine)
    if label is None:
        return None, None
    try:
        return label, raw.decode(label, "strict").lstrip("\ufeff")
    except (UnicodeError, LookupError):
        return label, None


def worker(engine: str):
    expected = {"codeccat": "0.1.0", "chardet": "7.7.0",
                "charset_normalizer": "3.5.2"}
    assert engine in expected
    assert importlib.metadata.version(
        {"charset_normalizer": "charset-normalizer"}.get(engine, engine)
    ) == expected[engine]
    if engine == "codeccat":
        import codeccat
        from importlib.resources import files
        assert digest(files("codeccat").joinpath("models/default.ccm").read_bytes()) == MODEL_SHA
        instance = codeccat.CodecCat()
    else:
        instance = None

    manifest = json.loads((WORK / "manifest.json").read_text())
    measurements, durations, cpus = [], [], []
    for row in manifest[:3]:
        detector_output(engine, (WORK / row["file"]).read_bytes(), instance)
    for row in manifest:
        data = (WORK / row["file"]).read_bytes()
        if row["pool"] == "natural_utf8":
            truth = data.decode("utf-8", "strict")
        elif row["pool"] == "cjk_natural_preexamined":
            name = row["source"].removesuffix(".txt")
            truth = (CPY / (name + "-utf8.txt")).read_bytes().decode("utf-8")
        else:
            # Recover original Unicode from strict, preregistered codec;
            # never consult model predictions to derive the reference.
            truth = data.decode(row["encoding"], "strict")
        assert digest(truth.encode("utf-8")) == row["truth_sha256"]
        start_wall, start_cpu = time.perf_counter_ns(), time.process_time_ns()
        label, decoded = detector_output(engine, data, instance)
        elapsed = (time.perf_counter_ns() - start_wall) / 1e6
        cpu = (time.process_time_ns() - start_cpu) / 1e6
        durations.append(elapsed)
        cpus.append(cpu)
        measurements.append({
            "pool": row["pool"], "group": row["group"], "encoding": row["encoding"],
            "size": row["size"], "source": row["source"],
            "correct_text": decoded == truth,
            "correct_label": label == row["encoding"],
            "silent_wrong": decoded is not None and decoded != truth,
            "abstained": decoded is None,
        })
    by_cohort = {}
    for pool in sorted(set(r["pool"] for r in measurements)):
        cases = [r for r in measurements if r["pool"] == pool]
        by_cohort[pool] = {
            "n": len(cases), "documents": len(set(r["group"] for r in cases)),
            "correct_text": sum(r["correct_text"] for r in cases),
            "correct_label": sum(r["correct_label"] for r in cases),
            "silent_wrong": sum(r["silent_wrong"] for r in cases),
            "abstained": sum(r["abstained"] for r in cases),
            "by_codec": {name: {
                "n": sum(r["encoding"] == name for r in cases),
                "correct_text": sum(r["correct_text"] for r in cases if r["encoding"] == name),
                "silent_wrong": sum(r["silent_wrong"] for r in cases if r["encoding"] == name),
            } for name in sorted({r["encoding"] for r in cases})},
        }
    output = {
        "engine": engine, "n": len(measurements), "by_cohort": by_cohort,
        "p50_ms": round(statistics.median(durations), 4),
        "p95_ms": round(sorted(durations)[int(.95 * (len(durations) - 1))], 4),
        "total_cpu_ms": round(sum(cpus), 3),
        "rss_peak_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "measured_calls": len(durations), "timing_method": "single sequential warm-cached process",
        "case_metadata": measurements,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{engine}.json").write_text(json.dumps(output, indent=2) + "\n")
    print("OVERALL_WORKER", engine, json.dumps({
        "n": output["n"], "by_cohort": {
            k: {p: v[p] for p in ("n", "correct_text", "silent_wrong", "abstained")}
            for k, v in by_cohort.items()
        }, "p50_ms": output["p50_ms"], "p95_ms": output["p95_ms"],
        "rss_peak_mib": output["rss_peak_mib"],
    }, sort_keys=True))


def summary():
    names = ("codeccat", "chardet", "charset_normalizer")
    data = {k: json.loads((RESULTS / f"{k}.json").read_text()) for k in names}
    pools = sorted(data["codeccat"]["by_cohort"])
    report = {
        "schema": "codeccat-stratified-benchmark-v1",
        "status": "completed",
        "versions": {"codeccat": "0.1.0", "chardet": "7.7.0",
                     "charset-normalizer": "3.5.2"},
        "sources": {
            "GNOME/gnome-terminal": "d11cd622f07760e0281841d6b084a0d9e62484a6",
            "python/cpython": "f64e51563a85667446793ad61ea521d005ceb171",
            "chardet/test-data": "b0c0d206913ba118ffea2f7d306883e7a4a76db7",
        },
        "selection": json.loads((WORK / "selection.json").read_text()),
        "by_pool": {p: {
            k: data[k]["by_cohort"][p] for k in names
        } for p in pools},
        "resources": {k: {f: data[k][f] for f in (
            "n", "p50_ms", "p95_ms", "total_cpu_ms", "rss_peak_mib",
        )} for k in names},
        "descriptive_equal_pool_average": {
            k: round(sum(
                data[k]["by_cohort"][p]["correct_text"] /
                data[k]["by_cohort"][p]["n"] for p in pools
            ) / len(pools) * 100, 2) for k in names
        },
        "warnings": [
            "Balanced pool average is not natural population accuracy.",
            "Western/CJK samples are multiple synthetic recodings from shared documents.",
            "CPython tests were examined earlier; no newly blind claim.",
            "Raw-byte overlap screen cannot prove semantic separation.",
            "Timing and peak RSS are single GitHub-hosted Linux process observations.",
        ],
        "raw_content_exported": False,
    }
    (RESULTS / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("OVERALL_STRATIFIED_SUMMARY", json.dumps({
        "by_pool": {pool: {engine: {
            "n": report["by_pool"][pool][engine]["n"],
            "correct_text": report["by_pool"][pool][engine]["correct_text"],
            "silent_wrong": report["by_pool"][pool][engine]["silent_wrong"],
        } for engine in names} for pool in pools},
        "resources": report["resources"],
        "balanced_descriptive_score": report["descriptive_equal_pool_average"],
    }, sort_keys=True))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("mode", choices=("prepare", "worker", "summary"))
    args.add_argument("--engine", choices=("codeccat", "chardet", "charset_normalizer"))
    opts = args.parse_args()
    if opts.mode == "prepare":
        prepare()
    elif opts.mode == "worker":
        if not opts.engine:
            raise ValueError("--engine required")
        worker(opts.engine)
    else:
        summary()
