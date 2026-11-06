"""Test external-corpus overlap screening on synthetic examples only."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[3] / "datasets" / "official" / "audit_holdout_overlap.py"
_SPEC = importlib.util.spec_from_file_location("holdout_overlap", _SOURCE)
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


class HoldoutOverlapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "training"
        self.gtk = self.root / "gtk"
        self.uch = self.root / "uch"
        for path in (self.data, self.gtk, self.uch / "samples"):
            path.mkdir(parents=True)
        self.raw = ("café résumé " * 700).encode("utf-8")
        (self.data / "a.bin").write_bytes(self.raw)
        manifest = [{
            "path": "a.bin",
            "encoding": "utf-8",
            "group": "training-a",
            "split": "train",
            "sha256": hashlib.sha256(self.raw).hexdigest(),
            "prefix_sha256": hashlib.sha256(self.raw[:4096]).hexdigest(),
        }]
        self.manifest = self.data / "manifest.jsonl"
        self.manifest.write_text(json.dumps(manifest) + "\n")

    def test_full_and_prefix_exact_hits(self):
        (self.uch / "samples" / "utf-8.txt").write_bytes(self.raw)
        result = module.audit(self.manifest, self.gtk, self.uch)
        stats = result["source_results"]["uchardet"]
        self.assertGreaterEqual(stats["exact_full_matches"], 1)
        self.assertGreaterEqual(stats["exact_prefix_matches"], 1)
        self.assertGreaterEqual(stats["candidate_near_overlap_windows"], 1)
        self.assertFalse(result["contains_original_text"])
        self.assertNotIn("café", json.dumps(result))

    def test_shared_blocks_screen_overlap_on_shifted_window(self):
        (self.uch / "samples" / "utf-8.txt").write_bytes(
            b"Preface to independent sample " * 4 + self.raw[250:2600]
        )
        result = module.audit(self.manifest, self.gtk, self.uch)
        stats = result["source_results"]["uchardet"]
        self.assertEqual(stats["exact_full_matches"], 0)
        self.assertEqual(stats["exact_prefix_matches"], 0)
        self.assertGreaterEqual(stats["candidate_near_overlap_windows"], 1)

    def test_mismatch_in_training_source_fails_closed(self):
        (self.data / "a.bin").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "training corpus hash mismatch"):
            module.audit(self.manifest, self.gtk, self.uch)

    def test_short_or_unrelated_inputs_are_not_marked_exact(self):
        (self.uch / "samples" / "utf-8.txt").write_bytes(
            b"entirely different contents " * 10
        )
        result = module.audit(self.manifest, self.gtk, self.uch)
        stats = result["source_results"]["uchardet"]
        self.assertEqual(stats["exact_full_matches"], 0)
        self.assertEqual(stats["exact_prefix_matches"], 0)


if __name__ == "__main__":
    unittest.main()
