"""Evaluation must reject split leakage before loading any candidate models."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from codeccat_train.evaluate import evaluate


class EvaluationIsolationTests(unittest.TestCase):
    def _run(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.bin").write_bytes(b"alpha")
            (root / "b.bin").write_bytes(b"bravo")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            fake = types.ModuleType("codeccat")
            class NoModelLoad:
                @classmethod
                def from_files(cls, *_args, **_kwargs):
                    raise AssertionError("model loaded before validating manifest")
            fake.CodecCat = NoModelLoad
            with patch.dict(sys.modules, {"codeccat": fake}):
                return evaluate(manifest, root / "not-needed.ccm", root / "result.json")

    def test_identical_bytes_cannot_cross_train_and_test(self):
        rows = [
            {"path": "a.bin", "encoding": "utf-8", "split": "train", "group": "g1"},
            {"path": "a.bin", "encoding": "utf-8", "split": "test", "group": "g2"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicated bytes across splits"):
            self._run(rows)

    def test_source_group_cannot_cross_splits_with_different_bytes(self):
        rows = [
            {"path": "a.bin", "encoding": "utf-8", "split": "train", "group": "shared"},
            {"path": "b.bin", "encoding": "utf-8", "split": "test", "group": "shared"},
        ]
        with self.assertRaisesRegex(ValueError, "source-group leakage"):
            self._run(rows)

    def test_unselected_split_cannot_hide_unsafe_path(self):
        rows = [
            {"path": "../outside.bin", "encoding": "utf-8", "split": "train", "group": "g1"},
            {"path": "a.bin", "encoding": "utf-8", "split": "test", "group": "g2"},
        ]
        with self.assertRaisesRegex(ValueError, "path escapes"):
            self._run(rows)


if __name__ == "__main__":
    unittest.main()
