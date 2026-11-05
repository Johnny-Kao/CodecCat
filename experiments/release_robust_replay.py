"""Frozen chardet/test-data release-wheel replay.

Reuses the two historical corpus-selection rules verbatim in effect:
- full-file digest dedup from vnext_strong_l2_dev.collect (historical 3,117)
- 4KB-prefix digest dedup from vnext_actual_adapter.corpus (historical 3,086)
Outputs summary and sample fingerprints only; never uploads raw training files.
This is a development corpus used by the release model. NOT a blind test.
"""
from __future__ import annotations

import codecs
import hashlib
import importlib.metadata
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import chardet
from charset_normalizer import from_bytes
from codeccat import CodecCat

ROOT = Path("training")
OUT = Path("research-results/native-release-robust-replay")
EXPECTED_REV = "b0c0d206913ba118ffea2f7d306883e7a4a76db7"


def canon(label):
    if not label:
        return None
    try:
        return codecs.lookup(label).name
    except (LookupError, TypeError):
        return None


def decode(raw, label, final):
    if not label:
        return None
    try:
        return codecs.getincrementaldecoder(label)("strict").decode(raw, final=final)
    except (LookupError, UnicodeError, TypeError):
        return None


def load_rows():
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_test_data import parse_dir_name

    rows = []
    full_seen = set()
    for path in sorted(ROOT.glob("*/*")):
        if not path.is_file():
            continue
        enc, _ = parse_dir_name(path.parent.name)
        truth = canon(enc)
        if not truth:
            continue
        original = path.read_bytes()
        if not original:
            continue
        try:
            original.decode(truth, "strict")
        except (LookupError, UnicodeError):
            continue
        full_sha = hashlib.sha256(original).hexdigest()
        if full_sha in full_seen:
            continue
        full_seen.add(full_sha)
        prefix = original[:4096]
        prefix_sha = hashlib.sha256(prefix).hexdigest()
        group = (path.stem.lower().replace("culturax_mc4_", "culturax_")
                 .replace("culturax_oscar-2301_", "culturax_"))
        rows.append({
            "raw": prefix,
            "truth": truth,
            "source": path.relative_to(ROOT).as_posix(),
            "full_sha": full_sha,
            "prefix_sha": prefix_sha,
            "group": group,
            "final": len(prefix) < 4096,
            "size": len(prefix),
        })
    return rows


def main():
    version = importlib.metadata.version("codeccat")
    assert version == "0.1.0a3", version
    detector = CodecCat()
    rows = load_rows()
    assert len(rows) >= 3000, f"unexpected frozen corpus count: {len(rows)}"

    start = time.perf_counter()
    for idx, row in enumerate(rows):
        raw = row["raw"]
        actual = decode(raw, row["truth"], row["final"])
        assert actual is not None, f"truncated prefix decoding failed: {row['source']}"

        # Same prefix-completeness semantics as historical development CV.
        # This calls the installed release's native-backed inference object.
        native = detector._model.rank(raw, final=row["final"])
        cdet = chardet.detect(raw).get("encoding")
        cn = from_bytes(raw).best()
        guesses = {
            "native_v0.1.0a3": native.encoding,
            "chardet_7.7.0": cdet,
            "charset_normalizer_3.5.2": cn.encoding if cn else None,
        }
        result = {}
        for name, prediction in guesses.items():
            norm = canon(prediction)
            pred_text = decode(raw, norm, row["final"])
            result[name] = {
                "exact_label": norm == row["truth"],
                "same_text": pred_text is not None and pred_text == actual,
                "prediction": norm,
            }
        row["result"] = result
        if (idx + 1) % 500 == 0:
            print("ROBUST_REPLAY_PROGRESS", idx + 1, len(rows), flush=True)

    unique = []
    seen = set()
    for row in rows:
        if row["prefix_sha"] in seen:
            continue
        seen.add(row["prefix_sha"])
        unique.append(row)

    def report(subset):
        result = {
            "n": len(subset),
            "exact_labels": {},
            "same_decoded_text": {},
            "short_256": sum(r["size"] <= 256 for r in subset),
            "full_prefix_4096": sum(not r["final"] for r in subset),
        }
        for name in ("native_v0.1.0a3", "chardet_7.7.0", "charset_normalizer_3.5.2"):
            exact = sum(r["result"][name]["exact_label"] for r in subset)
            same = sum(r["result"][name]["same_text"] for r in subset)
            result["exact_labels"][name] = {"correct": exact, "pct": round(exact * 100 / len(subset), 3)}
            result["same_decoded_text"][name] = {"correct": same, "pct": round(same * 100 / len(subset), 3)}
        return result

    # Report two versions of the *same pinned source corpus*. Do not invent
    # six additional rows to force a historical 3,092 denominator.
    digest = hashlib.sha256("\n".join(
        f"{r['full_sha']}:{r['truth']}:{r['group']}" for r in rows
    ).encode()).hexdigest()
    summary = {
        "schema": "codeccat-release-robust-replay-v1",
        "release": "v0.1.0a3",
        "model_fingerprint": detector.model_fingerprint,
        "source": "chardet/test-data",
        "source_commit": EXPECTED_REV,
        "selection": {
            "full_document_sha_dedup": report(rows),
            "prefix_4k_sha_dedup": report(unique),
        },
        "dataset_sha256": digest,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "warning": (
            "Training-corpus diagnostic, not an independent generalization "
            "accuracy test. The historical 94.243% used a different research "
            "model and fold-specific validation; this uses the shipped "
            "full-corpus-fitted native model. No 3,092-case fixture "
            "selection was identified in the recovered source."
        ),
        "api_scope": (
            "Native model .rank(..., final=len(prefix)<4096) is used to "
            "preserve historical prefix semantics. This is NOT an ordinary "
            "call to the public detect() API on complete documents. "
            "Competitors are decoded with the same incremental strict rule."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("ROBUST_RELEASE_REPLAY_RESULT_START", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("ROBUST_RELEASE_REPLAY_RESULT_END", flush=True)


if __name__ == "__main__":
    main()
