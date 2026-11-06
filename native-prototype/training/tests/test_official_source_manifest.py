"""Pinned source-manifest preparation without copying third-party corpus text."""
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[3] / "datasets" / "official" / "prepare_manifest.py"
_SPEC = importlib.util.spec_from_file_location("official_manifest_generator", _SOURCE)
generator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(generator)


class PinnedManifestTests(unittest.TestCase):
    def test_codec_directory_disambiguation(self):
        self.assertEqual(generator.encoding_from_dir("utf-16-be"), "utf-16")
        self.assertEqual(generator.encoding_from_dir("utf-16be-ru"), "utf-16-be")
        self.assertEqual(generator.encoding_from_dir("utf-32-be"), "utf-32")
        self.assertEqual(generator.encoding_from_dir("cp1252-fr"), "cp1252")
        self.assertEqual(generator.encoding_from_dir("cp1252-french"), "cp1252")
        self.assertIsNone(generator.encoding_from_dir("scripts"))

    def test_order_prefix_deduplication_and_hashes(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            latin = root / "cp1252-fr"
            latin.mkdir()
            first = b"A" * 4096 + b"first tail"
            second = b"A" * 4096 + b"another tail"
            (latin / "a.bin").write_bytes(first)
            (latin / "b.bin").write_bytes(second)
            (latin / "c.bin").write_bytes("résumé".encode("cp1252"))
            utf16 = root / "utf-16-be"
            utf16.mkdir()
            (utf16 / "doc.bin").write_bytes("test".encode("utf-16"))
            rows = generator.generate_records(root)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["path"], "cp1252-fr/a.bin")
            self.assertEqual(rows[1]["path"], "cp1252-fr/c.bin")
            self.assertEqual(rows[2]["encoding"], "utf-16")
            self.assertEqual(rows[0]["prefix_sha256"], hashlib.sha256(first[:4096]).hexdigest())
            self.assertEqual(rows[0]["sha256"], hashlib.sha256(first).hexdigest())
            self.assertTrue(all(r["split"] == "train" for r in rows))
            self.assertTrue(all(set(r) >= {"path","sha256","prefix_sha256","group","encoding"} for r in rows))

    def test_pinned_checkout_required_and_manifest_is_metadata_only(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            path = root / "cp1252-fr"
            path.mkdir()
            (path / "a.bin").write_bytes("résumé".encode("cp1252"))
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "cp1252-fr/a.bin"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                            "-c", "user.email=test@example.invalid", "commit", "-qm",
                            "fixture"], check=True)
            sha = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
            out = root / "codeccat-replay-manifest.jsonl"
            with self.assertRaisesRegex(ValueError,"wrong source commit"):
                generator.prepare(root,out,require_official_size=False)
            result = generator.prepare(root,out,expected_commit=sha,require_official_size=False)
            self.assertEqual(result["record_count"],1)
            self.assertFalse(result["contains_raw_text"])
            manifest = json.loads(out.read_text())
            self.assertEqual(manifest["encoding"],"cp1252")
            self.assertNotIn("résumé",out.read_text())
            (path / "a.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"tracked corpus files are modified"):
                generator.prepare(root,out,expected_commit=sha,require_official_size=False)


if __name__ == "__main__":
    unittest.main()
