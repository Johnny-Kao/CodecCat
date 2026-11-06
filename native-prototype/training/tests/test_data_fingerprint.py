"""The source fingerprint must change when input bytes change without manifest edits."""
import json
import tempfile
import unittest
from pathlib import Path

from codeccat_train.__main__ import fingerprint_dataset, read_manifest


class DataFingerprintTests(unittest.TestCase):
    def test_content_changes_under_same_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "train-a.bin").write_bytes(b"sample A")
            (base / "train-b.bin").write_bytes("café".encode("cp1252"))
            (base / "val.bin").write_bytes(b"validation")
            entries = [
                {"path":"train-a.bin","encoding":"utf-8","split":"train","group":"a"},
                {"path":"train-b.bin","encoding":"cp1252","split":"train","group":"b"},
                {"path":"val.bin","encoding":"utf-8","split":"validation","group":"c"},
            ]
            manifest = base / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(r)+"\n" for r in entries))
            before = fingerprint_dataset(sum(read_manifest(manifest), []))
            self.assertEqual(len(before), 64)
            self.assertEqual(before, fingerprint_dataset(sum(read_manifest(manifest), [])))
            (base / "train-a.bin").write_bytes(b"sample A changed")
            after = fingerprint_dataset(sum(read_manifest(manifest), []))
            self.assertNotEqual(before, after)

    def test_assignment_changes_fingerprint(self):
        rows = [(b"a", "utf-8", "group-a", "train", "sha-a")]
        other = [(b"a", "utf-8", "group-b", "train", "sha-a")]
        self.assertNotEqual(fingerprint_dataset(rows), fingerprint_dataset(other))


if __name__ == "__main__":
    unittest.main()
