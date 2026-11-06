"""Train-only residual replay is explicit and does not fabricate validation."""
import json
import tempfile
import unittest
from pathlib import Path

from codeccat_train.__main__ import fit


class ReplayFitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        records = []
        for n in range(24):
            for encoding, text in (
                ("utf-16-be", f"Valid structural sample {n:03d} for replay"),
                ("cp1252", f"Résumé numéro {n:03d} for replay"),
            ):
                name = f"{encoding}-{n}.bin"
                (self.root / name).write_bytes(text.encode(encoding))
                records.append({
                    "path": name,
                    "encoding": encoding,
                    "split": "train",
                    "group": f"source-{encoding}-{n}",
                })
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )

    def test_explicit_replay_without_validation_scores(self):
        output = self.root / "model.ccm"
        report = self.root / "report.json"
        result = fit(self.manifest, output, report, seed=20261015,
                     precision="f32", residual_focus=True, train_only=True)
        self.assertEqual(output.read_bytes()[:4], b"CCM2")
        self.assertEqual(result["training_original_n"], 48)
        self.assertTrue(result["residual_focus_applied"])
        self.assertGreater(sum(result["training_removed_by_label"].values()), 0)
        self.assertLess(result["training_n"], result["training_original_n"])
        self.assertEqual(result["validation_n"], 0)
        self.assertIsNone(result["validation_codec_label_accuracy"])
        self.assertTrue(result["train_only"])
        self.assertEqual(len(result["dataset_content_fingerprint_sha256"]), 64)
        self.assertEqual(len(result["effective_training_fingerprint_sha256"]), 64)
        self.assertEqual(json.loads(report.read_text()), result)

    def test_regular_fit_still_requires_validation(self):
        with self.assertRaisesRegex(ValueError, "validation records required"):
            fit(self.manifest, self.root / "model.ccm", self.root / "report.json",
                seed=20261015)


if __name__ == "__main__":
    unittest.main()
