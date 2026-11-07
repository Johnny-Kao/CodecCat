"""Rights inventory parses pinned upstream catalog; it never grants licenses."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_SCRIPT=Path(__file__).resolve().parents[3]/"datasets"/"official"/"rights_inventory.py"
_SPEC=importlib.util.spec_from_file_location("rights_inventory",_SCRIPT)
inventory=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inventory)


class RightsInventoryTests(unittest.TestCase):
    def test_catalog_source_and_provenance(self):
        catalog="""# Test catalog

#### `utf-8-japanese/` — 2 files

| File | Source | Provenance | Size | Notes |
|------|--------|------------|-----:|-------|
| `culturax_mC4_1.txt` | CulturaX | transcoded | 1,123 | |
| `legacy.xml` | chardet | wild | 2,151 | |

#### `cp1252-french/` — 1 files

| File | Source | Provenance | Size | Notes |
|------|--------|------------|-----:|-------|
| `fixture.txt` | uchardet | suite | 2,222 | |
"""
        records=inventory.catalog_index(catalog)
        self.assertEqual(records["utf-8-japanese/culturax_mC4_1.txt"],("CulturaX","transcoded"))
        self.assertEqual(records["utf-8-japanese/legacy.xml"],("chardet","wild"))
        self.assertEqual(records["cp1252-french/fixture.txt"],("uchardet","suite"))

    def test_audit_retains_uncleared_rights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=root/"manifest.jsonl"
            manifest.write_text(json.dumps({
                "path":"utf-8-japanese/legacy.xml",
                "sha256":"a"*64,"prefix_sha256":"b"*64
            })+"\n", encoding="utf-8")
            catalog=root/"CATALOG.md"
            catalog.write_text("#### `utf-8-japanese/` — 1 files\n\n| File | Source | Provenance | Size | Notes |\n|------|--------|------------|-----:|-------|\n| `legacy.xml` | chardet | wild | 2200 | |\n", encoding="utf-8")
            summary=inventory.audit(manifest,catalog,root/"result")
            self.assertEqual(summary["training_records"],1)
            self.assertEqual(summary["rights_review_required_records"],1)
            self.assertEqual(summary["training_rights_verified_records"],0)
            self.assertFalse(summary["release_authorized"])
            data=json.loads((root/"result"/"rights-inventory-ledger.jsonl").read_text())
            self.assertEqual(data["training_rights_status"],"UNVERIFIED")
            self.assertEqual(data["model_distribution_rights_status"],"UNVERIFIED")

    def test_fails_when_catalog_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=root/"manifest.jsonl"
            manifest.write_text(json.dumps({"path":"unknown/file.txt"})+"\n")
            catalog=root/"CATALOG.md"
            catalog.write_text("#### `utf-8/` — 1 files\n| `known.txt` | test | wild | 1 | |\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"absent from the pinned catalog"):
                inventory.audit(manifest,catalog,root/"result")


if __name__=="__main__":
    unittest.main()
