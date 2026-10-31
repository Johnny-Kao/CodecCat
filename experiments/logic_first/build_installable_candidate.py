"""Build the real alpha default model from an immutable checked-out corpus.

The result is a usable fitted model for an alpha package, not a claim of
superiority over competitors. All fitting runs in GitHub Actions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_l3_language import LanguageCoherence
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    actual = subprocess.run(
        ["git", "-C", args.corpus, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().lower()
    if actual != args.training_commit.lower():
        raise ValueError("training source is not at the pinned commit")
    rows = corpus(Path(args.corpus))
    if len(rows) < 1000 or len({label for _, label, _, _, _ in rows}) < 40:
        raise RuntimeError("insufficient verified model training corpus")
    training = [(b, label, group) for b, label, group, _, _ in rows]
    focused, removed = residual_subset(training)
    model = TrainableDetector.fit(focused, feature_profile="stable23")
    # L3 remains off by default. Ship the alpha candidate only as a
    # consciously chosen opt-in model that users may explicitly request.
    language = LanguageCoherence.fit(training)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out/"stable23.zip"
    language_path = out/"language-l3.json"
    model.save_portable(model_path)
    language.save(language_path)
    model_again = TrainableDetector.load_portable(model_path)
    language_again = LanguageCoherence.load(language_path)
    if model_again.fingerprint != model.fingerprint or language_again != language:
        raise RuntimeError("installed model portable round trip failed")
    manifest = {
        "schema": "codeccat-alpha-bundled-model-v1",
        "status": "alpha: callable model, not independently approved for accuracy superiority",
        "training_origin": "chardet/test-data",
        "training_commit": actual,
        "training_original_samples": len(rows),
        "training_residual_samples": len(focused),
        "training_removed_by_label": removed,
        "classes": len(model.classifier.classes_),
        "model": {"path": "stable23.zip", "sha256": sha(model_path),
                  "fingerprint": model.fingerprint},
        "optional_l3": {"path": "language-l3.json", "sha256": sha(language_path),
                        "fingerprint": language.fingerprint, "enabled": False},
        "licensing_note": "Model provenance originates in the pinned training dataset; review applicable fixture origins before external redistribution.",
    }
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print("USABLE_ALPHA_MODEL_BEGIN")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("USABLE_ALPHA_MODEL_END")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--training-commit", required=True)
    p.add_argument("--output", default="src/codeccat/models")
    main(p.parse_args())
