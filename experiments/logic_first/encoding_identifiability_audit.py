"""Development-only encoding-identifiability audit; never changes model weights.

A codec's distinguishable byte positions are an upper bound on byte-level evidence,
NOT proof that natural language semantics can identify the original codec.
"""
from __future__ import annotations
import argparse
import codecs
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus

PAIRS = (
    ("cp037", "cp500"),
    ("iso8859-16", "iso8859-2"),
    ("iso8859-16", "cp1250"),
    ("iso8859-2", "cp1250"),
    ("cp1252", "iso8859-1"),
    ("cp1252", "iso8859-15"),
    ("cp1252", "hp-roman8"),
)

def decode(data, encoding):
    try:
        return codecs.decode(data, encoding, errors="strict")
    except (UnicodeError, LookupError, ValueError):
        return None

def byte_differences(a, b):
    different = []
    for byte in range(256):
        x = decode(bytes((byte,)), a)
        y = decode(bytes((byte,)), b)
        if x != y:
            different.append(byte)
    return different

def summarize_pair(rows, a, b):
    difference = set(byte_differences(a, b))
    records = []
    for raw, label, group, fold, digest in rows:
        if label not in (a, b):
            continue
        truth = decode(raw, label)
        alternative = decode(raw, b if label == a else a)
        # The actual inference model uses at most 4096 input bytes.
        examined = raw[:4096]
        alt_prefix = decode(examined, b if label == a else a)
        truth_prefix = decode(examined, label)
        # Incomplete multi-byte prefixes are not evidence of impossibility.
        mapped_diff = sum(byte in difference for byte in examined)
        records.append({
            "sha": digest, "label": label, "group": group, "fold": fold,
            "size": len(raw), "prefix_size": len(examined),
            "discriminating_byte_count": mapped_diff,
            "has_discriminating_byte": bool(mapped_diff),
            "alternative_strict_decodable_full": alternative is not None,
            "exactly_same_text_full": truth is not None and alternative == truth,
            "exactly_same_text_prefix": truth_prefix is not None and alt_prefix == truth_prefix,
        })
    by_label = {}
    for label in (a, b):
        subset = [r for r in records if r["label"] == label]
        by_label[label] = {
            "n": len(subset),
            "groups": len({r["group"] for r in subset}),
            "alternate_decodable": sum(r["alternative_strict_decodable_full"] for r in subset),
            "same_text_full": sum(r["exactly_same_text_full"] for r in subset),
            "same_text_prefix": sum(r["exactly_same_text_prefix"] for r in subset),
            "zero_discriminating_bytes": sum(not r["has_discriminating_byte"] for r in subset),
        }
    return {
        "pair": [a, b],
        "different_single_byte_mappings": len(difference),
        "different_byte_values_hex": [f"{x:02x}" for x in sorted(difference)],
        "by_truth_label": by_label,
        "records": records,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--out", default="research-results/encoding-identifiability")
    args = parser.parse_args()
    rows = corpus(Path(args.fixtures))
    assert rows
    summary = [summarize_pair(rows, a, b) for a, b in PAIRS]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "codeccat-encoding-identifiability-v1",
        "status": "development-only-not-blind-acceptance",
        "source": "pinned chardet/test-data; training-origin corpus, not independent",
        "dataset_count": len(rows),
        "notes": [
            "Same-text bytes make the encoding label unidentifiable from that input alone.",
            "Different single-byte maps indicate potential evidence, not reliable natural-language attribution.",
            "No prediction improvement or model training is claimed.",
            "Prefix equality is conservative: unfinished multibyte prefixes may fail strict decoding.",
            "This is a prioritized diagnostic, not a statistically independent evaluation.",
        ],
        "pairs": [{k: v for k, v in pair.items() if k != "records"} for pair in summary],
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with (output / "cases.jsonl").open("w") as f:
        for pair in summary:
            for record in pair["records"]:
                f.write(json.dumps({"pair": pair["pair"], **record}, sort_keys=True) + "\n")
    print("IDENTIFIABILITY_AUDIT_JSON_START")
    print(json.dumps(metadata, indent=2))
    print("IDENTIFIABILITY_AUDIT_JSON_END")

if __name__ == "__main__":
    main()
