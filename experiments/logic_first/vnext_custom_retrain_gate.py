"""Research-only, reusable custom-corpus retraining and comparison gate.

The user supplies bytes, labels, document groups, source/rights evidence and
PREDEFINED training/validation partition. This tool never creates releases,
changes main, publishes packages or approves a candidate for deployment.

Example manifest.jsonl:
{"path":"train/01.bin","encoding":"utf-8","source_group":"author-a-doc-1","split":"train","origin":"my-files","rights":"owned"}
{"path":"validate/01.bin","encoding":"cp1252","source_group":"author-b-doc-9","split":"validation","origin":"my-files","rights":"owned"}

This is a developer research protocol, not a finished public-facing API.
"""
from __future__ import annotations

import argparse
import codecs
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_bom import inspect_bom


@dataclass(frozen=True)
class Sample:
    data: bytes
    label: str
    group: str
    split: str
    origin: str
    rights: str
    digest: str
    text_digest: str


def truth_text(data: bytes, label: str) -> str:
    text = data.decode(label, "strict")
    bom = inspect_bom(data, final=True)
    if bom.signature and text.startswith("\ufeff"):
        text = text[1:]
    return text



def suspicious_singlebyte_utf8(data: bytes, declared_codec: str) -> dict | None:
    """Identify a *possible* stale single-byte label using raw-byte evidence.

    A single-byte legacy codec can decode many invalidly tagged files without
    error. Only flag strong evidence of UTF-8-as-Latin mojibake; this does NOT
    prove the original historical codec and never relabels the training text.
    """
    if not isinstance(data, bytes):
        raise TypeError("expected bytes")
    if not (declared_codec.startswith("iso8859-") or
            declared_codec.startswith("cp12") or
            declared_codec in ("latin-1", "latin_1")):
        return None
    if len(data) < 64:
        return None
    try:
        candidate = data.decode("utf-8", "strict")
        declared = data.decode(declared_codec, "strict")
    except (UnicodeError, LookupError):
        return None
    non_ascii = sum(ord(c) > 127 for c in candidate)
    artifact_pairs = sum(
        a in ("\u00c2", "\u00c3") and 128 <= ord(b) <= 191
        for a, b in zip(declared, declared[1:])
    )
    if non_ascii < 4 or artifact_pairs < 3:
        return None
    return {
        "utf8_non_ascii": non_ascii,
        "mojibake_pairs_under_declared_codec": artifact_pairs,
        "reason": "possible_utf8_data_with_stale_singlebyte_encoding_label",
    }


def _resolved_data_file(root: Path, name: str) -> Path:
    if not name or Path(name).is_absolute() or "\\" in name:
        raise ValueError("manifest path must be a relative POSIX data path")
    bits = name.split("/")
    if any(bit in ("", ".", "..") for bit in bits):
        raise ValueError("manifest path must not traverse directories")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("manifest file is missing or escapes dataset root")
    return path


def load_manifest(root: Path, manifest: Path, *, max_bytes: int = 1 << 20) -> tuple[list[Sample], dict]:
    root = root.resolve()
    if not root.is_dir() or not manifest.is_file():
        raise ValueError("dataset root and manifest must exist")
    records: list[Sample] = []
    seen_bytes: dict[str, str] = {}
    used_path = set()
    reviewed_suspect_labels = []
    for i, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"manifest line {i}: invalid JSON") from e
        required = ("path", "encoding", "source_group", "split", "origin", "rights")
        if not isinstance(row, dict) or any(
            not isinstance(row.get(k), str) or not row[k].strip() for k in required
        ):
            raise ValueError(f"manifest line {i}: missing path/encoding/group/split/origin/rights")
        split = row["split"]
        if split not in ("train", "validation"):
            raise ValueError(f"manifest line {i}: unexpected split {split}")
        f = _resolved_data_file(root, row["path"])
        if f in used_path:
            raise ValueError(f"manifest line {i}: file listed twice")
        used_path.add(f)
        size = f.stat().st_size
        if size < 1 or size > max_bytes:
            raise ValueError(f"manifest line {i}: file size outside 1..{max_bytes}")
        data = f.read_bytes()
        try:
            label = codecs.lookup(row["encoding"]).name
            decoded = truth_text(data, label)
        except (LookupError, UnicodeError, ValueError) as e:
            raise ValueError(f"manifest line {i}: invalid labeled byte stream") from e
        digest = sha256(data).hexdigest()
        suspicion = suspicious_singlebyte_utf8(data, label)
        if suspicion is not None:
            if row.get("charset_provenance_reviewed") is not True:
                raise ValueError(
                    f"manifest line {i}: possible UTF-8 data with stale single-byte encoding label "
                    f"({label}); review original source and set charset_provenance_reviewed=true "
                    "ONLY with independent provenance evidence"
                )
            reviewed_suspect_labels.append({
                "path": row["path"], "sha256": digest, "declared_label": label,
                **suspicion,
                "verification": "SELF_ATTESTED_NOT_INDEPENDENTLY_VERIFIED",
            })
        if digest in seen_bytes:
            raise ValueError(f"manifest line {i}: repeated identical raw bytes")
        seen_bytes[digest] = label
        text_digest = sha256(decoded.encode("utf-8", "surrogatepass")).hexdigest()
        records.append(Sample(data, label, row["source_group"].strip(), split,
                              row["origin"].strip(), row["rights"].strip(),
                              digest, text_digest))
    if not records:
        raise ValueError("empty manifest")
    groups = {split: {r.group for r in records if r.split == split}
              for split in ("train", "validation")}
    assert len(groups) == 2
    if not all(groups.values()) or not groups["train"].isdisjoint(groups["validation"]):
        raise ValueError("train and validation must contain disjoint nonempty source groups")
    ids = {split: {r.text_digest for r in records if r.split == split}
           for split in ("train", "validation")}
    if not ids["train"].isdisjoint(ids["validation"]):
        raise ValueError("train and validation share the same decoded document content")
    train_labels = Counter(r.label for r in records if r.split == "train")
    if len(train_labels) < 2 or min(train_labels.values()) < 2:
        raise ValueError("training requires at least two classes and two rows per class")
    ledger = {
        "n_train": sum(r.split == "train" for r in records),
        "n_validation": sum(r.split == "validation" for r in records),
        "groups_train": len(groups["train"]),
        "groups_validation": len(groups["validation"]),
        "training_classes": dict(sorted(train_labels.items())),
        "validation_classes": dict(sorted(Counter(r.label for r in records if r.split == "validation").items())),
        "origin_counts": {f"{source} :: {rights}": n for (source, rights), n in sorted(Counter((r.origin, r.rights) for r in records).items())},
        "rights": "SELF-ATTESTED: owner must verify provenance, ownership and permission to train/distribute",
        "suspected_charset_mismatch_reviewed": reviewed_suspect_labels,
        "suspected_charset_review_rule": "default reject unless each flagged manifest row has explicit charset_provenance_reviewed=true",
    }
    return records, ledger


def detect(model: TrainableDetector, row: Sample) -> dict:
    # Strict, complete-object scoring; do not equate valid decode with truth.
    from codeccat.vnext_decoding import decode
    selected = decode(model, row.data, bom_policy="strip")
    known_text = truth_text(row.data, row.label)
    return {
        "chosen": selected.encoding,
        "label_hit": selected.encoding == row.label,
        "text_hit": selected.text is not None and selected.text == known_text,
        "accepted": selected.accepted,
        "ambiguous": selected.ambiguous,
        "reason": selected.reason,
    }


def encoding_family(name: str) -> str:
    if name.startswith("utf-"):
        return "unicode"
    if name.startswith(("iso8859-", "iso-8859-")):
        return "iso8859"
    if name.startswith("cp12"):
        return "windows"
    if name in ("cp037", "cp500", "cp273", "cp1140", "cp1026"):
        return "ebcdic"
    return "other"


def evaluate(model: TrainableDetector, rows: list[Sample]) -> tuple[dict, list[dict]]:
    cases = []
    for row in rows:
        pred = detect(model, row)
        cases.append({"sha": row.digest, "group": row.group, "truth": row.label,
                      "family": encoding_family(row.label), **pred})
    n = len(cases)
    by_family = {}
    for family in sorted({c["family"] for c in cases}):
        subset = [c for c in cases if c["family"] == family]
        by_family[family] = {
            "n": len(subset),
            "text_correct": sum(c["text_hit"] for c in subset),
            "label_correct": sum(c["label_hit"] for c in subset),
            "abstained": sum(not c["accepted"] for c in subset),
        }
    return {
        "n": n,
        "text_correct": sum(c["text_hit"] for c in cases),
        "label_correct": sum(c["label_hit"] for c in cases),
        "abstained": sum(not c["accepted"] for c in cases),
        "reported_ambiguous": sum(c["ambiguous"] for c in cases),
        "by_family": by_family,
    }, cases


def compare(before: list[dict], after: list[dict]) -> dict:
    assert len(before) == len(after)
    assert all(a["sha"] == b["sha"] for a, b in zip(before, after))
    paired = {
        "helps": sum(b["text_hit"] and not a["text_hit"] for a, b in zip(before, after)),
        "harms": sum(a["text_hit"] and not b["text_hit"] for a, b in zip(before, after)),
    }
    paired["net"] = paired["helps"] - paired["harms"]
    group = defaultdict(lambda: {"n": 0, "old": 0, "new": 0})
    for a, b in zip(before, after):
        item = group[a["family"]]
        item["n"] += 1
        item["old"] += int(a["text_hit"])
        item["new"] += int(b["text_hit"])
    return {
        "paired_decoded_text": paired,
        "per_family": dict(sorted(group.items())),
        "families_regressing_in_correct_text": [
            f for f, x in group.items() if x["new"] < x["old"]
        ],
    }


def run(args):
    rows, ledger = load_manifest(Path(args.root), Path(args.manifest))
    training = [(r.data, r.label, r.group) for r in rows if r.split == "train"]
    holdout = [r for r in rows if r.split == "validation"]
    if args.policy == "residual":
        from vnext_residual_study import residual_subset
        training, dropped = residual_subset(training)
    else:
        dropped = {}
    candidate = TrainableDetector.fit(training, feature_profile="stable23")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "candidate.ccnext"
    candidate.save_portable(archive)
    restored = TrainableDetector.load_portable(archive)
    if restored.fingerprint != candidate.fingerprint:
        raise RuntimeError("candidate portable archive round-trip mismatch")
    checks, cases = evaluate(restored, holdout)
    if [x["text_hit"] for x in cases] != [x["text_hit"] for x in evaluate(candidate, holdout)[1]]:
        raise RuntimeError("portable model changes validation decisions")

    compare_result = None
    if args.incumbent:
        baseline = TrainableDetector.load_portable(args.incumbent)
        past, prior_cases = evaluate(baseline, holdout)
        compare_result = {
            "incumbent": past,
            "difference": compare(prior_cases, cases),
            "incumbent_sha256": sha256(Path(args.incumbent).read_bytes()).hexdigest(),
        }
    # This tool deliberately does not treat a favorable comparison on a
    # repeatedly-observed development set as permission to publish a model.
    # An external, untouched confirmation set is a separate mandatory gate.
    result = {
        "schema": "codeccat-custom-retrain-quality-v1",
        "decision": "RESEARCH_CANDIDATE_ONLY",
        "promotion_allowed": False,
        "reason": "an independently sourced and untouched final acceptance set is required",
        "manifest_sha256": sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "training_record_sha256": sha256(
            "".join(r.digest + r.label + r.group + "\n" for r in rows if r.split == "train").encode()
        ).hexdigest(),
        "source_ledger": ledger,
        "training_policy": args.policy,
        "removed_from_training_by_label": dropped,
        "model": {
            "feature_profile": "stable23",
            "schema": "numeric-only-portable",
            "sha256": sha256(archive.read_bytes()).hexdigest(),
            "fingerprint": restored.fingerprint,
            "n_classes": len(restored.classifier.classes_),
        },
        "validation": checks,
        "paired_with_incumbent": compare_result,
        "limitations": [
            "External source and rights are self-attested, not independently verified.",
            "The fixed validation split is a developer selection set, NOT final acceptance.",
            "Near-duplicate or semantically reused documents may escape exact-content checks.",
            "No confidence threshold, performance target or production approval is inferred.",
        ],
    }
    (out / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "validation.jsonl").write_text("\n".join(json.dumps(c, sort_keys=True) for c in cases) + "\n")
    print("CUSTOM_RETRAIN_GATE_JSON_START", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("CUSTOM_RETRAIN_GATE_JSON_END", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="research-results/custom-retrain-gate")
    p.add_argument("--policy", choices=("full", "residual"), default="residual")
    p.add_argument("--incumbent", help="optional previously fitted portable .ccnext model")
    run(p.parse_args())
