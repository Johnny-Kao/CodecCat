"""Fail-closed validation of the historical Stable23 L2 + optional L3 replay.

Reads only aggregate JSON reports from the manual research workflow. It never
loads source text or uploads model artifacts. A successful reproduction is
distinct from rights clearance, quality evaluation, or release authorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

L2_SHA256 = "16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b"
L3_SHA256 = "2eae47abe3674e258f54ce500f75ac7abf1d86b8a706fe6ba004273ed3c1ecfd"
SOURCE_COMMIT = "b0c0d206913ba118ffea2f7d306883e7a4a76db7"
SOURCE_MANIFEST_SHA256 = "8ec3007dbd59336cee6018cfa44aeded39cca205e93f5fc9f24ee5ecf87f3ca9"
L2_TRAINING_FINGERPRINT = "936eb0331a0d0ac447b0f6c6135bec7cd397962544d053cda3220358dacba85c"
L3_TRAINING_FINGERPRINT = "ab085534af6e5fb518ab46c9dd2a40bba941761d591b8bcaafff85bc94e53fa3"


def _check(label: str, actual, expected, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def validate(root: Path) -> dict:
    """Raise ValueError on any drift, including unknown or missing metadata."""
    l2 = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    l3 = json.loads((root / "l3-comparison.json").read_text(encoding="utf-8"))
    numeric = json.loads((root / "numeric-comparison.json").read_text(encoding="utf-8"))
    l2_training = json.loads((root / "training-report.json").read_text(encoding="utf-8"))
    l3_training = json.loads((root / "l3-report.json").read_text(encoding="utf-8"))
    l3_repeat = json.loads((root / "l3-repeat-report.json").read_text(encoding="utf-8"))

    errors: list[str] = []
    checks = [
        ("L2 source commit", l2.get("source_commit"), SOURCE_COMMIT),
        ("L2 input manifest", l2.get("input_manifest_sha256"), SOURCE_MANIFEST_SHA256),
        ("L2 training original samples", l2.get("training_original_n"), 3086),
        ("L2 training selected samples", l2.get("training_effective_n"), 2799),
        ("L2 training inputs", l2.get("training_inputs_match_reference"), True),
        ("L2 training fingerprint", l2.get("candidate_training_fingerprint"), L2_TRAINING_FINGERPRINT),
        ("L2 official training fingerprint", l2.get("official_training_fingerprint"), L2_TRAINING_FINGERPRINT),
        ("L2 reference SHA", l2.get("official_ccm2_sha256"), L2_SHA256),
        ("L2 candidate SHA", l2.get("candidate_ccm2_sha256"), L2_SHA256),
        ("L2 repeat SHA", l2.get("repeat_ccm2_sha256"), L2_SHA256),
        ("L2 byte equality", l2.get("byte_identical_model"), True),
        ("L2 same-run determinism", l2.get("same_runner_replays_identical"), True),
        ("L2 model schema", l2_training.get("model_schema"), "CCM2"),
        ("L2 weight precision", l2_training.get("weight_precision"), "f32"),
        ("L2 source manifest", l2_training.get("manifest_sha256"), SOURCE_MANIFEST_SHA256),
        ("L2 selected samples in report", l2_training.get("training_n"), 2799),
        ("L2 original samples in report", l2_training.get("training_original_n"), 3086),
        ("L2 validation records", l2_training.get("validation_n"), 0),
        ("L2 train-only mode", l2_training.get("train_only"), True),
        ("L2 residual selection", l2_training.get("residual_focus_applied"), True),
        ("L2 model SHA report", l2_training.get("model_sha256"), L2_SHA256),
        ("L2 reported fingerprint", l2_training.get("historical_stable23_fingerprint"), L2_TRAINING_FINGERPRINT),
        ("L2 numeric identity", numeric.get("identical_files"), True),
        ("L2 numeric alignment", numeric.get("aligned_for_numeric_comparison"), True),
        ("L2 vocabulary identity", numeric.get("same_vocabulary"), True),
        ("L2 vocab set identity", numeric.get("same_vocabulary_set"), True),
        ("L2 reference features", numeric.get("reference_features"), 45000),
        ("L2 candidate features", numeric.get("candidate_features"), 45000),
        ("L2 reference classes", numeric.get("reference_classes"), 88),
        ("L2 candidate classes", numeric.get("candidate_classes"), 88),
        ("L2 candidate hash in numeric comparison", numeric.get("candidate_sha256"), L2_SHA256),
        ("L3 candidate SHA", l3.get("candidate_sha256"), L3_SHA256),
        ("L3 reference SHA", l3.get("reference_sha256"), L3_SHA256),
        ("L3 byte identity", l3.get("byte_identical_model"), True),
        ("L3 fingerprint identity", l3.get("same_fingerprint"), True),
        ("L3 trigram identity", l3.get("same_trigram_counts"), True),
        ("L3 count identity", l3.get("same_total"), True),
        ("L3 alpha identity", l3.get("same_alpha"), True),
        ("L3 candidate gram count", l3.get("candidate_gram_count"), 30000),
        ("L3 reference gram count", l3.get("reference_gram_count"), 30000),
        ("L3 default disabled", l3.get("optional_l3_remains_disabled_by_default"), True),
        ("L3 report schema", l3_training.get("schema"), "vnext-l3-unicode-trigram-v1"),
        ("L3 source manifest", l3_training.get("input_manifest_sha256"), SOURCE_MANIFEST_SHA256),
        ("L3 input samples", l3_training.get("input_n"), 3086),
        ("L3 training fingerprint", l3_training.get("training_fingerprint"), L3_TRAINING_FINGERPRINT),
        ("L3 model SHA report", l3_training.get("model_sha256"), L3_SHA256),
        ("L3 repeat model SHA report", l3_repeat.get("model_sha256"), L3_SHA256),
        ("L3 repeat training fingerprint", l3_repeat.get("training_fingerprint"), L3_TRAINING_FINGERPRINT),
        ("L3 default off", l3_training.get("optional_model_enabled_by_default"), False),
    ]
    for label, actual, expected in checks:
        _check(label, actual, expected, errors)

    for name in ("idf", "intercept", "weights"):
        stat = numeric.get(name)
        if not isinstance(stat, dict):
            errors.append(f"L2 {name}: missing numeric tensor comparison")
        else:
            _check(f"L2 {name} changed values", stat.get("changed_values"), 0, errors)

    if errors:
        raise ValueError("historical replay NOT IDENTICAL:\n" + "\n".join(errors))

    return {
        "status": "REPRODUCTION_PASS_NOT_RELEASE_AUTHORIZATION",
        "l2_sha256": L2_SHA256,
        "l3_sha256": L3_SHA256,
        "source_commit": SOURCE_COMMIT,
        "checked_invariants": len(checks) + 3,
        "rights_review_complete": False,
        "release_approval": False,
        "l3_enabled_by_default": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.reports)
    print("STABLE23_STRICT_REPLAY_PASS", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
