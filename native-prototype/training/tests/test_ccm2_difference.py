"""CCM2 comparison reports numeric differences without model text."""
import hashlib
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

_SOURCE=Path(__file__).resolve().parents[3]/"datasets"/"official"/"compare_ccm.py"
_SPEC=importlib.util.spec_from_file_location("ccm2_numeric_compare",_SOURCE)
module=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


def make_ccm(path,weights,vocabulary=("ab","bc"),classes=("utf-8","cp1252","shift_jis")):
    parts=[struct.pack("<4sIIBBBB",b"CCM2",len(vocabulary),len(classes),2,3,2,0)]
    for name in classes:
        encoded=name.encode()
        parts.append(struct.pack("<H",len(encoded))+encoded)
    for index,gram in enumerate(vocabulary):
        encoded=gram.encode()
        parts.append(struct.pack("<I",index)+struct.pack("<H",len(encoded))+encoded)
    parts.append(struct.pack("<"+"d"*len(vocabulary),*[1.0]*len(vocabulary)))
    rows=1 if len(classes)==2 else len(classes)
    parts.append(struct.pack("<"+"d"*rows,*[0.5]*rows))
    parts.append(struct.pack("<"+"f"*len(weights),*weights))
    body=b"".join(parts)
    path.write_bytes(body+hashlib.sha256(body).digest())


class Ccm2NumericDifferenceTests(unittest.TestCase):
    def test_numeric_delta_and_feature_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"reference.ccm",Path(tmp)/"candidate.ccm"
            make_ccm(a,[1.0,2.0,3.0,4.0,5.0,6.0])
            make_ccm(b,[1.0,2.25,3.0,4.0,5.0,6.0])
            result=module.compare(a,b)
            self.assertTrue(result["aligned_for_numeric_comparison"])
            self.assertFalse(result["identical_files"])
            self.assertEqual(result["idf"]["changed_values"],0)
            self.assertEqual(result["intercept"]["changed_values"],0)
            self.assertEqual(result["weights"]["changed_values"],1)
            self.assertAlmostEqual(result["weights"]["max_abs_delta"],0.25)
            self.assertEqual(result["top_classes_by_weight_delta"][0]["encoding"],"utf-8")
            self.assertNotIn("\'ab\'",str(result))

    def test_mismatched_vocabulary_blocks_numeric_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a.ccm",Path(tmp)/"b.ccm"
            make_ccm(a,[1.0]*6)
            make_ccm(b,[1.0]*6,vocabulary=("ab","cd"))
            result=module.compare(a,b)
            self.assertFalse(result["same_vocabulary"])
            self.assertFalse(result["same_vocabulary_set"])
            self.assertEqual(result["shared_features"],1)
            self.assertEqual(result["reference_only_features"],1)
            self.assertEqual(result["candidate_only_features"],1)
            self.assertFalse(result["aligned_for_numeric_comparison"])
            self.assertEqual(result["idf_on_shared_features"]["changed_values"],0)
            self.assertNotIn("weights",result)

    def test_index_permutation_aligns_same_grams(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a.ccm",Path(tmp)/"b.ccm"
            make_ccm(a,[1.0,2.0,3.0,4.0,5.0,6.0],vocabulary=("ab","bc"))
            make_ccm(b,[2.0,1.0,4.0,3.0,6.0,5.0],vocabulary=("bc","ab"))
            result=module.compare(a,b)
            self.assertTrue(result["same_vocabulary_set"])
            self.assertFalse(result["same_vocabulary"])
            self.assertEqual(result["shared_features_with_different_index"],2)
            self.assertEqual(result["weights_on_shared_features"]["changed_values"],0)
            self.assertEqual(result["idf_on_shared_features"]["changed_values"],0)

    def test_binary_classifier_one_weight_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a.ccm",Path(tmp)/"b.ccm"
            make_ccm(a,[1.0,2.0],classes=("utf-8","cp1252"))
            make_ccm(b,[1.0,2.0],classes=("utf-8","cp1252"))
            result=module.compare(a,b)
            self.assertTrue(result["identical_files"])
            self.assertEqual(result["weights"]["values"],2)
            self.assertEqual(len(result["top_classes_by_weight_delta"]),1)

    def test_rejects_corrupted_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/"bad.ccm"
            make_ccm(a,[1.0]*6)
            corrupted=bytearray(a.read_bytes());corrupted[20]^=1
            a.write_bytes(corrupted)
            with self.assertRaisesRegex(ValueError,"digest mismatch"):
                module.load_ccm2(a)


if __name__ == "__main__":
    unittest.main()
