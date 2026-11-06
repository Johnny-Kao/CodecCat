"""Compare the standalone Trainer with the original Stable23 L2 learner.

Synthetic examples only. This checks equivalent learning mathematics, not
reproduction of the bundled official model or its corpus.
"""
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

# The reference learner is only a development-time test oracle; the released
# Trainer package does not import the repository's legacy/source Runtime.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
from codeccat.vnext_strong_l2 import TrainableDetector

from codeccat_train.__main__ import fit_l2, historical_stable23_fingerprint


def samples():
    rows = []
    for index in range(24):
        for encoding, phrase in (
            ("utf-8", f"日本語の分析と文字コードの検証 {index}"),
            ("cp1252", f"café résumé français numéro {index}"),
            ("shift_jis", f"日本語の符号化処理と資料 {index}"),
        ):
            data = phrase.encode(encoding)
            rows.append((data, encoding, f"source-{index}-{encoding}"))
    return rows


class Stable23NumericalParityTests(unittest.TestCase):
    def test_same_examples_produce_same_learning_parameters(self):
        training = samples()
        reference = TrainableDetector.fit(training, feature_profile="stable23")
        rows = [
            (raw, encoding, group, "train", hashlib.sha256(raw).hexdigest())
            for raw, encoding, group in training
        ]
        vectorizer, classifier = fit_l2(rows, seed=20261015)
        self.assertEqual(historical_stable23_fingerprint(rows), reference.fingerprint)

        self.assertEqual(vectorizer.vocabulary_, reference.vectorizer.vocabulary_)
        self.assertEqual(vectorizer.ngram_range, reference.vectorizer.ngram_range)
        self.assertEqual(list(classifier.classes_), list(reference.classifier.classes_))
        np.testing.assert_array_equal(vectorizer.idf_, reference.vectorizer.idf_)
        np.testing.assert_array_equal(classifier.intercept_, reference.classifier.intercept_)
        np.testing.assert_array_equal(classifier.coef_, reference.classifier.coef_)

    def test_reference_and_trainer_use_same_4096_byte_prefix(self):
        training = [
            (raw + b" " * 5000, encoding, group)
            for raw, encoding, group in samples()
        ]
        reference = TrainableDetector.fit(training, feature_profile="stable23")
        rows = [
            (raw, encoding, group, "train", hashlib.sha256(raw).hexdigest())
            for raw, encoding, group in training
        ]
        vectorizer, classifier = fit_l2(rows, seed=20261015)
        self.assertEqual(vectorizer.vocabulary_, reference.vectorizer.vocabulary_)
        np.testing.assert_array_equal(vectorizer.idf_, reference.vectorizer.idf_)
        np.testing.assert_array_equal(classifier.coef_, reference.classifier.coef_)


if __name__ == "__main__":
    unittest.main()
