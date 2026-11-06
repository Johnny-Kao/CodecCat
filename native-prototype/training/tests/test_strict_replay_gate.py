"""The manual reproduction gate must fail on L2 or L3 drift."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[3] / "datasets" / "official" / "replay_gate.py"
_SPEC = importlib.util.spec_from_file_location("strict_replay_gate", _SOURCE)
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


def fixtures():
    return {
        "comparison.json": {
            "source_commit": module.SOURCE_COMMIT,
            "input_manifest_sha256": module.SOURCE_MANIFEST_SHA256,
            "training_original_n": 3086, "training_effective_n": 2799,
            "training_inputs_match_reference": True,
            "candidate_training_fingerprint": module.L2_TRAINING_FINGERPRINT,
            "official_training_fingerprint": module.L2_TRAINING_FINGERPRINT,
            "official_ccm2_sha256": module.L2_SHA256,
            "candidate_ccm2_sha256": module.L2_SHA256,
            "repeat_ccm2_sha256": module.L2_SHA256,
            "byte_identical_model": True, "same_runner_replays_identical": True,
        },
        "training-report.json": {
            "model_schema": "CCM2", "weight_precision": "f32",
            "manifest_sha256": module.SOURCE_MANIFEST_SHA256,
            "training_n": 2799, "training_original_n": 3086,
            "validation_n": 0, "train_only": True, "residual_focus_applied": True,
            "model_sha256": module.L2_SHA256,
            "historical_stable23_fingerprint": module.L2_TRAINING_FINGERPRINT,
        },
        "numeric-comparison.json": {
            "identical_files": True, "aligned_for_numeric_comparison": True,
            "same_vocabulary": True, "same_vocabulary_set": True,
            "reference_features": 45000, "candidate_features": 45000,
            "reference_classes": 88, "candidate_classes": 88,
            "candidate_sha256": module.L2_SHA256,
            "idf": {"changed_values": 0},
            "intercept": {"changed_values": 0},
            "weights": {"changed_values": 0},
        },
        "l3-comparison.json": {
            "candidate_sha256": module.L3_SHA256,
            "reference_sha256": module.L3_SHA256,
            "byte_identical_model": True,
            "same_fingerprint": True,
            "same_trigram_counts": True,
            "same_total": True, "same_alpha": True,
            "candidate_gram_count": 30000, "reference_gram_count": 30000,
            "optional_l3_remains_disabled_by_default": True,
        },
        "l3-report.json": {
            "schema": "vnext-l3-unicode-trigram-v1",
            "input_manifest_sha256": module.SOURCE_MANIFEST_SHA256,
            "input_n": 3086,
            "training_fingerprint": module.L3_TRAINING_FINGERPRINT,
            "model_sha256": module.L3_SHA256,
            "optional_model_enabled_by_default": False,
        },
        "l3-repeat-report.json": {
            "model_sha256": module.L3_SHA256,
            "training_fingerprint": module.L3_TRAINING_FINGERPRINT,
        },
    }


class StrictReplayGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payload = fixtures()
        self.write()

    def write(self):
        for filename, data in self.payload.items():
            (self.root / filename).write_text(json.dumps(data))

    def test_clean_reproduction_passes_without_release_authorization(self):
        result = module.validate(self.root)
        self.assertEqual(result["status"], "REPRODUCTION_PASS_NOT_RELEASE_AUTHORIZATION")
        self.assertFalse(result["rights_review_complete"])
        self.assertFalse(result["release_approval"])

    def test_l2_sha_drift_is_failure(self):
        self.payload["comparison.json"]["candidate_ccm2_sha256"] = "0" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "L2 candidate SHA"):
            module.validate(self.root)

    def test_l3_sha_drift_is_failure(self):
        self.payload["l3-comparison.json"]["candidate_sha256"] = "0" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "L3 candidate SHA"):
            module.validate(self.root)

    def test_missing_l3_proof_is_failure(self):
        self.payload["l3-comparison.json"].pop("same_trigram_counts")
        self.write()
        with self.assertRaisesRegex(ValueError, "L3 trigram identity"):
            module.validate(self.root)

    def test_changed_weight_values_are_failure(self):
        self.payload["numeric-comparison.json"]["weights"]["changed_values"] = 1
        self.write()
        with self.assertRaisesRegex(ValueError, "L2 weights changed values"):
            module.validate(self.root)

    def test_train_validation_leakage_is_failure(self):
        self.payload["training-report.json"]["validation_n"] = 1
        self.write()
        with self.assertRaisesRegex(ValueError, "L2 validation records"):
            module.validate(self.root)

    def test_missing_file_fails(self):
        (self.root / "l3-repeat-report.json").unlink()
        with self.assertRaises(FileNotFoundError):
            module.validate(self.root)


if __name__ == "__main__":
    unittest.main()
