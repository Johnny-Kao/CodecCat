"""Identify model state/prediction mutation when fitting a second detector.

Compare old fit before and after fitting a separate, content-disjoint candidate.
No data or model files are modified in the repository. GitHub Actions only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

from codeccat.vnext_strong_l2 import TrainableDetector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_training_strategy_audit import strict_text
from vnext_fit_determinism_replay import fit_signature


def docs(data, label):
    return sha256(strict_text(data, label).encode("utf-8", "surrogatepass")).hexdigest()


def predictions(model, held):
    return {digest: str(model.rank(data, final=len(data) < 4096).encoding)
            for data, truth, group, digest in held}


def run(args):
    rows = corpus(Path(args.fixtures))
    results = []
    for fold in (0, 2):
        train = [(b, y, g) for b, y, g, k, _ in rows if k != fold]
        held = [(b, y, g, digest) for b, y, g, k, digest in rows if k == fold]
        known = {docs(b, y) for b, y, _, _ in held}
        old_training, _ = residual_subset(train)
        clean_training, _ = residual_subset([(b, y, g) for b, y, g in train if docs(b, y) not in known])
        old = TrainableDetector.fit(old_training, feature_profile="stable23")
        before_state = fit_signature(old)
        before = predictions(old, held)
        new = TrainableDetector.fit(clean_training, feature_profile="stable23")
        after_state = fit_signature(old)
        after = predictions(old, held)
        new_preds = predictions(new, held)
        changed = [sha for sha in before if before[sha] != after[sha]]
        report = {
            "fold": fold,
            "old_training": len(old_training),
            "new_training": len(clean_training),
            "old_model_before_numeric_hash": before_state,
            "old_model_after_numeric_hash": after_state,
            "old_prediction_changes_after_other_model_fit": len(changed),
            "examples_changed": changed[:12],
            "old_prediction_before_sha": sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
            "old_prediction_after_sha": sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
            "new_model_numeric_hash": fit_signature(new),
            "new_model_prediction_sha": sha256(json.dumps(new_preds, sort_keys=True).encode()).hexdigest(),
        }
        results.append(report)
        print("POSTFIT_ISOLATION_CASE", json.dumps(report, sort_keys=True), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    if any(r["old_prediction_changes_after_other_model_fit"] for r in results):
        raise AssertionError("training a second model changed earlier model predictions")
    if any(r["old_model_before_numeric_hash"] != r["old_model_after_numeric_hash"] for r in results):
        raise AssertionError("training a second model mutated earlier learned parameters")
    print("POSTFIT_ISOLATION_PASS", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", required=True)
    p.add_argument("--out", default="research-results/postfit-isolation")
    run(p.parse_args())
