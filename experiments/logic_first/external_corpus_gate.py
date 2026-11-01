"""Provenance and leakage gate for external CodecCat evaluation corpora.

Manifest JSONL records: path, encoding, source_id, document_id, split, license,
origin_url, verification_method. Only manifest-listed local files are read.
"""
from __future__ import annotations
import argparse
import codecs
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

REQUIRED = ("path", "encoding", "source_id", "document_id", "split", "license", "origin_url", "verification_method")
SPLITS = {"train", "validation", "confirmation"}
METHODS = {"source_metadata", "creator_attestation", "verified_original"}
def validate(manifest: Path, root: Path):
    errors, records, known_bytes = [], [], {}
    groups = defaultdict(set)
    seen_paths = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"ok": False, "errors": [str(exc)], "samples": 0}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, TypeError) as exc:
            errors.append(f"line {number}: invalid JSON: {exc}")
            continue
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in REQUIRED):
            errors.append(f"line {number}: missing/non-string mandatory field")
            continue
        if row["split"] not in SPLITS or row["verification_method"] not in METHODS:
            errors.append(f"line {number}: invalid split or verification method")
            continue
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"line {number}: unsafe relative path")
            continue
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            errors.append(f"line {number}: path escapes corpus or file is missing")
            continue
        if str(target) in seen_paths:
            errors.append(f"line {number}: repeated file path")
            continue
        seen_paths.add(str(target))
        try:
            codec = codecs.lookup(row["encoding"]).name
            raw = target.read_bytes()
            if not raw:
                raise ValueError("empty document")
            raw.decode(codec, errors="strict")
        except (LookupError, UnicodeError, ValueError, OSError) as exc:
            errors.append(f"line {number}: invalid encoding, file or strict decode: {exc}")
            continue
        key = hashlib.sha256(raw).hexdigest()
        if key in known_bytes and known_bytes[key] != row["split"]:
            errors.append(f"line {number}: exact-byte duplicate crosses splits")
        known_bytes.setdefault(key, row["split"])
        groups[("source", row["source_id"])].add(row["split"])
        groups[("document", row["document_id"])].add(row["split"])
        records.append({"encoding": codec, "split": row["split"], "sha256": key, "source_id": row["source_id"], "document_id": row["document_id"], "bytes": len(raw)})
    for (kind, identifier), splits in sorted(groups.items()):
        if len(splits) > 1:
            errors.append(f"{kind} {identifier!r}: crosses splits ({','.join(sorted(splits))})")
    counts = Counter(r["split"] for r in records)
    if not records:
        errors.append("empty valid corpus")
    if not counts["confirmation"]:
        errors.append("no untouched confirmation samples")
    return {"schema": "codeccat-provenance-gate-v1", "ok": not errors, "errors": errors,
            "samples": len(records), "by_split": dict(counts),
            "by_encoding": dict(sorted(Counter(r["encoding"] for r in records).items())),
            "source_count": len({r["source_id"] for r in records}),
            "document_count": len({r["document_id"] for r in records}),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "note": "Metadata and exact-byte checks only; authenticity, licenses, near duplicates and independence require separate review."}
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = validate(args.manifest, args.root)
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    raise SystemExit(0 if result["ok"] else 1)
if __name__ == "__main__":
    main()
