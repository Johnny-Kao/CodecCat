"""Optional L3 must match the original historical trainer byte-for-byte."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from codeccat_train.l3 import fit_l3, save_l3

_REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "historical_l3", _REPO / "src" / "codeccat" / "vnext_l3_language.py"
)
module = importlib.util.module_from_spec(_SPEC)
import sys
sys.modules[_SPEC.name] = module
_SPEC.loader.exec_module(module)


def samples():
    rows=[]
    for i in range(50):
        for codec, text in (
            ("utf-8", f"日本語の文章と資料の整理番号 {i:03d} 漢字とひらがな"),
            ("cp1252", f"Éléments de référence et données françaises numéro {i:03d}"),
        ):
            raw=text.encode(codec)
            rows.append((raw,codec,f"group-{i}-{codec}"))
    rows.append((("日本語の分析データ " * 600).encode("utf-8"),"utf-8","long-document"))
    return rows


class L3ParityTests(unittest.TestCase):
    def test_serialized_bytes_match_original(self):
        rows=samples()
        historical=module.LanguageCoherence.fit(
            [(raw[:4096],codec,group) for raw,codec,group in rows]
        )
        model=fit_l3(rows)
        self.assertEqual(model["grams"],historical.grams)
        self.assertEqual(model["total"],historical.total)
        self.assertEqual(model["fingerprint"],historical.fingerprint)
        with tempfile.TemporaryDirectory() as d:
            ref=Path(d)/"original.json"
            new=Path(d)/"new.json"
            historical.save(ref)
            report=save_l3(rows,new)
            self.assertEqual(ref.read_bytes(),new.read_bytes())
            self.assertEqual(report["model_sha256"],hashlib.sha256(ref.read_bytes()).hexdigest())

    def test_original_4kb_prefix_and_decode_gate(self):
        rows=samples()
        rows.append((b"bad\xff","utf-8","invalid-utf8"))
        expected=module.LanguageCoherence.fit(
            [(raw[:4096],codec,group) for raw,codec,group in rows]
        )
        got=fit_l3(rows)
        self.assertEqual(got["fingerprint"],expected.fingerprint)
        self.assertEqual(got["grams"],expected.grams)

    def test_rejects_empty_and_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError,"max_features"):
            fit_l3(samples(),max_features=99)
        with self.assertRaisesRegex(ValueError,"no decodable"):
            fit_l3([(b"\xff","utf-8","broken")])


if __name__=="__main__":
    unittest.main()
