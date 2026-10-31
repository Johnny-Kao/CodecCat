"""Paired experiment: effect of removing cross-fold decoded-document leakage.

Same heldout rows, same stable23 architecture, same residual training recipe.
The only changed input is *training* rows whose decoded document text is
present among validation examples. This is still development evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict
import sys

from codeccat.vnext_strong_l2 import TrainableDetector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_training_strategy_audit import strict_text, examine, family


def text_id(raw, label):
    s = strict_text(raw, label)
    assert s is not None
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()


def summarize(cases):
    rows = len(cases)
    result = {"n": rows}
    for name in ("original", "text_disjoint"):
        preds = [r[name] for r in cases]
        result[name] = {
            "correct_label": sum(x["label"] for x in preds),
            "correct_text": sum(x["text"] for x in preds),
            "abstain": sum(x["abstain"] for x in preds),
            "truth_in_top3": sum(x["truth_in_top3"] for x in preds),
        }
    result["paired"] = {
        metric: {
            "helps": sum(r["text_disjoint"][metric] and not r["original"][metric] for r in cases),
            "harms": sum(r["original"][metric] and not r["text_disjoint"][metric] for r in cases),
        } for metric in ("text", "label")
    }
    for v in result["paired"].values():
        v["net"] = v["helps"] - v["harms"]
    result["family"] = {}
    for fam in sorted({family(r["truth"]) for r in cases}):
        sub = [r for r in cases if family(r["truth"]) == fam]
        result["family"][fam] = {
            "n": len(sub),
            "original_text": sum(r["original"]["text"] for r in sub),
            "text_disjoint_text": sum(r["text_disjoint"]["text"] for r in sub),
        }
    return result


def run(args):
    rows = corpus(Path(args.fixtures))
    folds = []
    all_cases = []
    for fold in range(3):
        train = [(b, y, g) for b, y, g, k, _ in rows if k != fold]
        held = [(b, y, g, digest) for b, y, g, k, digest in rows if k == fold]
        held_text_ids = {text_id(b, y) for b, y, _, _ in held}
        clean = [(b, y, g) for b, y, g in train if text_id(b, y) not in held_text_ids]
        removed = Counter(y for b, y, g in train if text_id(b, y) in held_text_ids)
        assert len(clean) + sum(removed.values()) == len(train)
        assert {text_id(b, y) for b, y, _ in clean}.isdisjoint(held_text_ids)
        assert {g for _, _, g in clean}.isdisjoint({g for _, _, g, _ in held})
        # Original sampling logic is kept identical after removing the
        # accidentally duplicated content: no retuning of drop thresholds.
        old_residual, _ = residual_subset(train)
        clean_residual, _ = residual_subset(clean)
        model_old = TrainableDetector.fit(old_residual, feature_profile="stable23")
        model_clean = TrainableDetector.fit(clean_residual, feature_profile="stable23")
        original_training_text_ids = {text_id(b, y) for b, y, _ in train}
        results = []
        for data, truth, group, digest in held:
            ref_text = strict_text(data, truth)
            assert ref_text is not None
            results.append({
                "sha": digest, "truth": truth, "group": group, "fold": fold,
                "was_originally_exposed_to_training": text_id(data, truth) in original_training_text_ids,
                "original": examine(model_old, data, truth, ref_text),
                "text_disjoint": examine(model_clean, data, truth, ref_text),
            })
        all_cases.extend(results)
        folds.append({
            "fold": fold, "original_training": len(train),
            "text_disjoint_training": len(clean),
            "removed_train_rows": sum(removed.values()),
            "removed_by_label": dict(removed),
            "original_residual_training": len(old_residual),
            "text_disjoint_residual_training": len(clean_residual),
            "holdout": summarize(results),
        })
        print("LEAKAGE_REMOVAL_FOLD", fold, json.dumps({
            "removed_train": sum(removed.values()),
            "results": folds[-1]["holdout"],
        }), flush=True)
    report = {
        "schema": "codeccat-paired-document-overlap-removal-v1",
        "source": "chardet/test-data pinned; development-only",
        "status": "diagnostic-not-independent-proof",
        "model": "exact released stable23 runtime; identical residual fit policy in two arms",
        "control": "training retains cross-fold identical decoded content",
        "treatment": "remove from training any decoded text matching heldout; no other change",
        "aggregate": summarize(all_cases),
        "heldout_cases_originally_exposed": sum(r["was_originally_exposed_to_training"] for r in all_cases),
        "folds": folds,
        "limitations": [
            "Source groups and exact text hashes miss near-duplicate documents.",
            "Removed rows can alter classes/sample counts, not a single-feature intervention.",
            "No external corpus and no candidate model promotion.",
            "This experiment has been run on repeatedly observed development splits.",
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "paired.jsonl").write_text("\n".join(json.dumps(x, sort_keys=True) for x in all_cases) + "\n")
    print("LEAKAGE_REMOVAL_JSON_START", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("LEAKAGE_REMOVAL_JSON_END", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", required=True)
    p.add_argument("--out", default="research-results/leakage-removal")
    run(p.parse_args())
