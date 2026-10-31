"""Audit whether existing train/test folds isolate *documents*, not just filenames.

A strict encoding detector cannot identify a unique codec from bytes that
multiple codecs interpret as the same text. This is a diagnostic, not a
candidate model or an alpha release gate. Source data is pinned in CI.
"""
from __future__ import annotations

import argparse
import codecs
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus


PAIRS = (
    ("cp037", "cp500"),
    ("iso8859-16", "iso8859-2"),
    ("iso8859-2", "cp1250"),
    ("cp1252", "iso8859-1"),
    ("cp1252", "iso8859-15"),
)


def decode(raw: bytes, name: str) -> str | None:
    try:
        dec = codecs.getincrementaldecoder(name)("strict")
        return dec.decode(raw, final=len(raw) < 4096)
    except (UnicodeError, LookupError, ValueError):
        return None


def single_byte_difference(a: str, b: str) -> dict:
    disagreements = []
    gaps = []
    for v in range(256):
        aa = decode(bytes([v]), a)
        bb = decode(bytes([v]), b)
        if aa != bb:
            disagreements.append(v)
        if aa is None or bb is None:
            gaps.append(v)
    return {
        "different_byte_values": disagreements,
        "n_different_values": len(disagreements),
        "bytes_not_decodable_by_both": gaps,
    }


def metrics_pair(rows, a, b):
    mapping = single_byte_difference(a, b)
    different = set(mapping["different_byte_values"])
    counts = Counter()
    examples = []
    for raw, truth, group, fold, digest in rows:
        if truth not in (a, b):
            continue
        left = decode(raw, a)
        right = decode(raw, b)
        counts["total"] += 1
        counts["truth_" + truth] += 1
        counts["both_decode"] += (left is not None and right is not None)
        if left is None or right is None:
            counts["only_one_valid"] += 1
            continue
        identical = left == right
        counts["identical_decoded_text"] += identical
        contains = any(byte in different for byte in raw)
        counts["contains_distinguishing_byte"] += contains
        counts["different_decoded_text"] += not identical
        if not identical and len(examples) < 8:
            places = [i for i, (x, y) in enumerate(zip(left, right)) if x != y]
            examples.append({
                "sha": digest,
                "true_codec": truth,
                "source_group": group,
                "n_codepoint_differences": len(places),
                "first_difference_at": places[0] if places else None,
            })
    return {
        "first": a, "second": b, "mapping": mapping,
        "counts": dict(counts), "examples": examples,
        "interpretation": "Same decoded text means codec labels cannot be identified from this input. "
                          "Different text is a necessary but NOT sufficient condition to infer the historical label.",
    }


def components(rows):
    """Transitive document grouping over source_group and exact decoded text.

    The original source group name is kept even when the same decoded
    document was re-encoded under another label or renamed in test-data.
    """
    parent = list(range(len(rows)))
    size = [1] * len(rows)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    group_head, text_head = {}, {}
    text_folds = defaultdict(set)
    text_groups = defaultdict(set)
    text_cases = defaultdict(list)
    for i, (raw, label, group, fold, digest) in enumerate(rows):
        truth_text = decode(raw, label)
        assert truth_text is not None, (label, digest)
        key = hashlib.sha256(truth_text.encode("utf-8", "surrogatepass")).hexdigest()
        # Stable exact decoded content; no forced near-duplicate matching.
        if group in group_head:
            union(i, group_head[group])
        else:
            group_head[group] = i
        if key in text_head:
            union(i, text_head[key])
        else:
            text_head[key] = i
        text_folds[key].add(fold)
        text_groups[key].add(group)
        text_cases[key].append(i)

    rows_in_cross_fold_text = set()
    cross_fold_texts = 0
    cross_fold_groups = set()
    cross_source_texts = 0
    for key, inds in text_cases.items():
        if len(text_folds[key]) > 1:
            cross_fold_texts += 1
            rows_in_cross_fold_text.update(inds)
            cross_fold_groups.update(text_groups[key])
        if len(text_groups[key]) > 1:
            cross_source_texts += 1

    comp = defaultdict(list)
    for i in range(len(rows)):
        comp[find(i)].append(i)
    sizes = sorted((len(indices) for indices in comp.values()), reverse=True)
    assignments = {}
    for indices in comp.values():
        # Deterministic 3-way bin-packing, preserving full connected components.
        key = min(rows[i][4] for i in indices)
        assignments[key] = indices
    loads = [0, 0, 0]
    fold_by_row = [None] * len(rows)
    for key, indices in sorted(assignments.items(), key=lambda item: (-len(item[1]), item[0])):
        k = min(range(3), key=lambda f: (loads[f], f))
        loads[k] += len(indices)
        for i in indices:
            fold_by_row[i] = k
    assert all(v is not None for v in fold_by_row)
    assert all(len({fold_by_row[i] for i in indices}) == 1 for indices in comp.values())

    altered = sum(fold_by_row[i] != rows[i][3] for i in range(len(rows)))
    distribution = []
    for fold in range(3):
        held = [rows[i] for i in range(len(rows)) if fold_by_row[i] == fold]
        training = [rows[i] for i in range(len(rows)) if fold_by_row[i] != fold]
        train_labels = {r[1] for r in training}
        distribution.append({
            "fold": fold,
            "size": len(held),
            "labels_in_heldout": len({x[1] for x in held}),
            "labels_unseen_in_training": dict(Counter(x[1] for x in held if x[1] not in train_labels)),
            "training_size": len(training),
        })
    cross_orig_fold = []
    for indices in comp.values():
        if len({rows[i][3] for i in indices}) > 1:
            cross_orig_fold.append(len(indices))
    return {
        "n": len(rows),
        "unique_source_groups": len(group_head),
        "unique_decoded_documents": len(text_cases),
        "exact_decoded_documents_in_multiple_filename_groups": cross_source_texts,
        "exact_decoded_documents_crossing_original_fold": cross_fold_texts,
        "rows_in_cross_fold_document_content": len(rows_in_cross_fold_text),
        "source_groups_touched_by_cross_fold_documents": len(cross_fold_groups),
        "content_connected_components": len(comp),
        "largest_component_sizes": sizes[:15],
        "components_crossing_original_folds": len(cross_orig_fold),
        "rows_in_crossing_components": sum(cross_orig_fold),
        "rows_reassigned_by_proposed_component_safe_folds": altered,
        "proposed_content_group_disjoint_folds": distribution,
        "caveat": "Exact decoded text detects duplicates, not paraphrases/near-duplicates; "
                  "group filename stems are only approximate document provenance.",
    }


def run(args):
    rows = corpus(Path(args.fixtures))
    records = components(rows)
    paired = [metrics_pair(rows, a, b) for a, b in PAIRS]
    report = {
        "schema": "codeccat-training-data-identity-and-label-ambiguity-audit-v1",
        "status": "development-diagnostic; no retraining, release or new model",
        "training_origin": "chardet/test-data at pinned Git revision",
        "sample_count": len(rows),
        "document_grouping": records,
        "codec_pairs": paired,
        "next_gate": "Do not promote a candidate based on filename-group folds "
                     "if decoded-document overlap spans those folds. "
                     "Use component-disjoint splits and a genuinely external heldout for promotion.",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print("DOCUMENT_IDENTITY_AUDIT_BEGIN")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("DOCUMENT_IDENTITY_AUDIT_END")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--out", default="research-results/document-identity-audit")
    run(parser.parse_args())
