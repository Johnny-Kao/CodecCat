"""Build source-provenance/rights-review ledger from a pinned upstream catalog.

This is not a license classifier: a catalog's source and provenance labels
do NOT establish training or derivative-model redistribution permission.
No source text is read, stored, or published by this inventory script.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

_DIR = re.compile(r"^#{2,6}\s+`([^`]+/)`")
_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")


def catalog_index(markdown: str) -> dict[str, tuple[str, str]]:
    current: str | None = None
    result: dict[str, tuple[str, str]] = {}
    for line in markdown.splitlines():
        heading = _DIR.match(line)
        if heading:
            current = heading.group(1)
            continue
        if line.startswith("#"):
            current = None
            continue
        record = _ROW.match(line)
        if record and current:
            filename, source, provenance = (
                record.group(1), record.group(2).strip(), record.group(3).strip()
            )
            key = current + filename
            if key in result:
                raise ValueError("duplicate catalog path: " + key)
            result[key] = (source, provenance)
    return result


def audit(manifest: Path, catalog: Path, destination: Path) -> dict:
    index = catalog_index(catalog.read_text(encoding="utf-8"))
    if not index:
        raise ValueError("catalog contains no per-file provenance entries")
    parsed = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not parsed:
        raise ValueError("empty training manifest")
    ledger = []
    missing = []
    by_source = Counter()
    by_provenance = Counter()
    for entry in parsed:
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("training manifest entry missing path")
        if path not in index:
            missing.append(path)
            continue
        source, provenance = index[path]
        by_source[source] += 1
        by_provenance[provenance] += 1
        ledger.append({
            "path": path,
            "full_sha256": entry["sha256"],
            "prefix_sha256": entry["prefix_sha256"],
            "catalog_source": source,
            "catalog_provenance": provenance,
            "training_rights_status": "UNVERIFIED",
            "model_distribution_rights_status": "UNVERIFIED",
        })
    if missing:
        raise ValueError(
            f"training manifest has {len(missing)} paths absent from the pinned catalog; "
            "cannot claim complete provenance mapping"
        )
    summary = {
        "schema": "codeccat-train-source-rights-inventory-v1",
        "source_repository": "chardet/test-data",
        "source_revision": "b0c0d206913ba118ffea2f7d306883e7a4a76db7",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "training_records": len(parsed),
        "matched_catalog_records": len(ledger),
        "by_source": dict(sorted(by_source.items())),
        "by_catalog_provenance": dict(sorted(by_provenance.items())),
        "training_rights_verified_records": 0,
        "model_distribution_rights_verified_records": 0,
        "rights_review_required_records": len(ledger),
        "legal_clearance_status": "NOT_DETERMINED",
        "release_authorized": False,
        "note": (
            "The upstream README says each file remains copyright its publisher. "
            "Provenance classifications and public repository visibility do not grant "
            "training or derived-model distribution rights. Per-source review is required."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"rights-inventory-summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    (destination/"rights-inventory-ledger.jsonl").write_text(
        "".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in ledger),
        encoding="utf-8"
    )
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--catalog",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    summary=audit(args.manifest,args.catalog,args.output_dir)
    print("RIGHTS_INVENTORY_SOURCE_COUNTS",json.dumps({
        "training_records":summary["training_records"],
        "by_source":summary["by_source"],
        "by_provenance":summary["by_catalog_provenance"],
        "legal_clearance_status":summary["legal_clearance_status"],
    },sort_keys=True))


if __name__=="__main__":
    main()
