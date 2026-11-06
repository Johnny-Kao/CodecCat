"""Audit exact and candidate near-overlap with historical L2 training prefixes.

This is a screening tool: it cannot prove semantic or source independence.
It reads pinned data in ephemeral CI and emits hashes/counts only; never text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_BLOCK = 128
_STEP = 64


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def blocks(raw: bytes) -> set[bytes]:
    """Hash sampled 128-byte spans; overlap candidates are not proof."""
    if len(raw) < _BLOCK:
        return set()
    return {
        hashlib.blake2b(raw[pos:pos + _BLOCK], digest_size=16).digest()
        for pos in range(0, len(raw) - _BLOCK + 1, _STEP)
    }


def gtk_samples(root: Path):
    """Original GTK PO documents and benchmark-like masked windows."""
    for path in sorted(root.glob("*.po")):
        raw = path.read_bytes()
        raw = re.sub(
            br"(?i)(charset=)[A-Za-z0-9_.+-]+",
            lambda m: m.group(1) + b"?" * (len(m.group(0)) - len(m.group(1))),
            raw,
        )
        for window, offset in enumerate((0, len(raw) // 3, 2 * len(raw) // 3)):
            piece = raw[offset:offset + 8192]
            if len(piece) >= 128 and not piece.isascii():
                yield path.name, window, piece


def uchardet_samples(root: Path):
    """Original BYVoid fixture bytes and prefixes, before codec exclusions."""
    for path in sorted(root.glob("*/*")):
        if not path.is_file() or path.name == "uchardet-tests.c":
            continue
        if path.suffix.lower() not in (".txt", ".smi", ".le", ".be"):
            continue
        raw = path.read_bytes()
        for take in sorted({len(raw), min(256, len(raw)), min(4096, len(raw))}):
            piece = raw[:take]
            if len(piece) >= 128:
                yield path.relative_to(root).as_posix(), take, piece


def audit(training_manifest: Path, gtk_root: Path, uchardet_root: Path, *,
          min_shared_blocks: int = 3) -> dict:
    root = training_manifest.resolve().parent
    training = [json.loads(line) for line in training_manifest.read_text().splitlines() if line.strip()]
    if not training:
        raise ValueError("empty training manifest")
    full_index = defaultdict(set)
    prefix_index = defaultdict(set)
    block_index = defaultdict(set)
    for record in training:
        rel = Path(record["path"])
        path = (root / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(root):
            raise ValueError("training path escapes corpus")
        raw = path.read_bytes()
        full = sha(raw)
        prefix = sha(raw[:4096])
        if full != record["sha256"] or prefix != record["prefix_sha256"]:
            raise ValueError("training corpus hash mismatch")
        source_id = record["path"]
        full_index[full].add(source_id)
        prefix_index[prefix].add(source_id)
        for block in blocks(raw[:4096]):
            block_index[block].add(source_id)

    by_source = {}
    for name, samples in (
        ("gtk", gtk_samples(gtk_root)),
        ("uchardet", uchardet_samples(uchardet_root)),
    ):
        n = exact_full = exact_prefix = near_candidates = 0
        matched_training = set()
        cases = []
        for source, window, raw in samples:
            n += 1
            full_hits = full_index.get(sha(raw), set())
            prefix_hits = prefix_index.get(sha(raw[:4096]), set())
            overlap_counts = Counter(
                candidate
                for block in blocks(raw)
                for candidate in block_index.get(block, ())
            )
            candidates = [
                candidate for candidate, count in overlap_counts.items()
                if count >= min_shared_blocks
            ]
            matched_training.update(full_hits)
            matched_training.update(prefix_hits)
            matched_training.update(candidates)
            exact_full += bool(full_hits)
            exact_prefix += bool(prefix_hits)
            near_candidates += bool(candidates)
            if full_hits or prefix_hits or candidates:
                cases.append({
                    "external_source": source,
                    "window": window,
                    "external_sha256": sha(raw),
                    "exact_full": bool(full_hits),
                    "exact_prefix": bool(prefix_hits),
                    "candidate_shared_blocks": max(overlap_counts.values(), default=0),
                    "training_file_count": len(set(full_hits) | set(prefix_hits) | set(candidates)),
                })
        by_source[name] = {
            "screened_windows": n,
            "exact_full_matches": exact_full,
            "exact_prefix_matches": exact_prefix,
            "candidate_near_overlap_windows": near_candidates,
            "unique_training_files_flagged": len(matched_training),
            "flagged": cases,
        }
    return {
        "schema": "codeccat-train-external-overlap-audit-v1",
        "training_records": len(training),
        "training_manifest_sha256": sha(training_manifest.read_bytes()),
        "screening_span_bytes": _BLOCK,
        "sampling_step_bytes": _STEP,
        "minimum_shared_spans_for_candidate": min_shared_blocks,
        "source_results": by_source,
        "interpretation": (
            "Exact matches establish literal overlap. Shared sampled spans only "
            "flag candidates; absence of matches cannot establish semantic independence "
            "or exclude unexamined source documents."
        ),
        "contains_original_text": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--gtk-root", type=Path, required=True)
    parser.add_argument("--uchardet-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    out = audit(args.training_manifest, args.gtk_root, args.uchardet_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    compact = {
        name: {key: value for key, value in stats.items() if key != "flagged"}
        for name, stats in out["source_results"].items()
    }
    print("HOLDOUT_OVERLAP_SCREEN", json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
