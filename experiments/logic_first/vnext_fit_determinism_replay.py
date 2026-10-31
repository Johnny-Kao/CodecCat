"""Investigate whether stable23 fits are reproducible across process/fit order.

Model fingerprint covers the INPUT rows only; this audit hashes the ACTUAL
learned vocabulary, IDF, class order, coefficients, intercept, predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
from codeccat.vnext_strong_l2 import TrainableDetector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def fit_signature(model):
    h = hashlib.sha256()
    vocabulary = sorted((str(k), int(v)) for k, v in model.vectorizer.vocabulary_.items())
    h.update(json.dumps(vocabulary, sort_keys=True).encode())
    h.update(model.vectorizer.idf_.tobytes())
    h.update(json.dumps([str(x) for x in model.classifier.classes_]).encode())
    h.update(np.ascontiguousarray(model.classifier.coef_).tobytes())
    h.update(np.ascontiguousarray(model.classifier.intercept_).tobytes())
    return h.hexdigest()


def run(args):
    rows = corpus(Path(args.fixtures))
    fold = args.fold
    training = [(b, y, g) for b, y, g, k, _ in rows if k != fold]
    held = [(b, y, g, digest) for b, y, g, k, digest in rows if k == fold]
    trimmed, _ = residual_subset(training)
    if args.fit_order == "full-first":
        TrainableDetector.fit(training, feature_profile="stable23")
    elif args.fit_order == "clean-first":
        from vnext_training_strategy_audit import strict_text
        ids = {
            sha_bytes(strict_text(data, label).encode("utf-8", "surrogatepass"))
            for data, label, _, _ in held
        }
        clean = [
            (b, y, g) for b, y, g in training
            if sha_bytes(strict_text(b, y).encode("utf-8", "surrogatepass")) not in ids
        ]
        reduced, _ = residual_subset(clean)
        TrainableDetector.fit(reduced, feature_profile="stable23")
    model = TrainableDetector.fit(trimmed, feature_profile="stable23")
    fp = sha_bytes("".join(sha_bytes(b) + y + g + "\n" for b, y, g in trimmed).encode())
    ranked = [model.rank(b, final=len(b) < 4096).encoding for b, y, g, d in held]
    predictions = {digest: str(p) if p is not None else None
                   for (b, y, g, digest), p in zip(held, ranked)}
    report = {
        "schema": "codeccat-learned-fit-replay-v1",
        "case": args.fit_order,
        "fold": fold,
        "samples_train": len(trimmed),
        "samples_heldout": len(held),
        "training_rows_sha": fp,
        "model_input_fingerprint": model.fingerprint,
        "learned_numeric_state_sha": fit_signature(model),
        "predictions_sha": sha_bytes(json.dumps(predictions, sort_keys=True).encode()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"fold{fold}-{args.fit_order}.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / f"fold{fold}-{args.fit_order}-predictions.json").write_text(json.dumps(predictions, sort_keys=True))
    print("MODEL_DETERMINISM_DIAGNOSTIC", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", required=True)
    p.add_argument("--out", default="research-results/determinism-replay")
    p.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--fit-order", choices=("residual-first", "full-first", "clean-first"), required=True)
    run(p.parse_args())
