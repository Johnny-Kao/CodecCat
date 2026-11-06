"""Boundary checks for community-provided training manifests."""
import json
import tempfile
import unittest
from pathlib import Path
from codeccat_train.__main__ import read_manifest


class ManifestSafetyTests(unittest.TestCase):
    def setUp(self):
        self.work=tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root=Path(self.work.name)
        (self.root/"a.bin").write_bytes("café été".encode("cp1252"))
        (self.root/"b.bin").write_bytes("日本語日本語".encode("utf-8"))

    def write(self, records):
        p=self.root/"manifest.jsonl"
        p.write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf-8")
        return p

    def base(self):
        return [
            {"path":"a.bin","encoding":"cp1252","split":"train","group":"a"},
            {"path":"b.bin","encoding":"utf-8","split":"validation","group":"b"},
        ]

    def test_reject_parent_traversal(self):
        rows=self.base()
        rows[0]["path"]="../escape.bin"
        with self.assertRaisesRegex(ValueError,"escapes manifest"):
            read_manifest(self.write(rows))

    def test_reject_group_leak(self):
        rows=self.base()
        rows[1]["group"]="a"
        with self.assertRaisesRegex(ValueError,"source-group leakage"):
            read_manifest(self.write(rows))

    def test_reject_wrong_encoding(self):
        rows=self.base()
        rows[1]["encoding"]="ascii"
        with self.assertRaisesRegex(ValueError,"label does not decode"):
            read_manifest(self.write(rows))

    def test_duplicate_does_not_hide_source_group_across_splits(self):
        rows = [
            {"path":"a.bin","encoding":"cp1252","split":"train","group":"original"},
            {"path":"a.bin","encoding":"cp1252","split":"train","group":"shared"},
            {"path":"b.bin","encoding":"utf-8","split":"validation","group":"shared"},
        ]
        with self.assertRaisesRegex(ValueError,"source-group leakage"):
            read_manifest(self.write(rows))

    def test_optional_content_sha256_rejects_modified_inputs(self):
        import hashlib
        rows=self.base()
        (self.root/"c.bin").write_bytes(b"separate utf8 training sample")
        rows.insert(1,{"path":"c.bin","encoding":"utf-8","split":"train","group":"c"})
        rows[0]["sha256"]=hashlib.sha256((self.root/"a.bin").read_bytes()).hexdigest()
        rows[0]["prefix_sha256"]=rows[0]["sha256"]
        path=self.write(rows)
        read_manifest(path)
        (self.root/"a.bin").write_bytes("changed café".encode("cp1252"))
        with self.assertRaisesRegex(ValueError,"sha256 mismatch"):
            read_manifest(path)

    def test_optional_prefix_sha256_rejects_changed_prefix(self):
        rows=self.base()
        rows[0]["prefix_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"prefix_sha256 mismatch"):
            read_manifest(self.write(rows))

    def test_reject_duplicate_bytes_cross_splits(self):
        rows=self.base()
        rows[1]["path"]="a.bin"
        rows[1]["encoding"]="cp1252"
        with self.assertRaisesRegex(ValueError,"duplicate bytes cross labels/splits"):
            read_manifest(self.write(rows))


if __name__=="__main__":
    unittest.main()
