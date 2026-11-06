"""Historical residual-selection parity without external training data."""
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "experiments" / "logic_first"))

from vnext_residual_study import (
    residual_subset as original_subset,
    stable_structural_guess as original_guess,
)
from codeccat_train.residual import (
    residual_subset as new_subset,
    stable_structural_guess as new_guess,
)


def rows():
    result = []
    for i in range(16):
        for label, raw in (
            ("utf-16-be", f"Alphabetical test number {i:03d}".encode("utf-16-be")),
            ("utf-16-le", f"Alphabetical test number {i:03d}".encode("utf-16-le")),
            ("utf-8-sig", f"Échantillon {i:03d}".encode("utf-8-sig")),
            ("cp1252", f"Résumé numéro {i:03d}".encode("cp1252")),
        ):
            result.append((raw, label, f"group-{i}-{label}"))
    return result


class ResidualReplayTests(unittest.TestCase):
    def test_exact_historical_selection_and_drop_counts(self):
        records = rows()
        selected, removed = new_subset(records)
        expected, expected_removed = original_subset(records)
        self.assertEqual(selected, expected)
        self.assertEqual(removed, expected_removed)
        self.assertGreater(sum(removed.values()), 0)
        self.assertEqual({x[1] for x in selected}, {x[1] for x in records})

    def test_structural_guess_matches_historical_rule(self):
        known = {r[1] for r in rows()}
        cases = [
            b"",
            b"hello",
            b"\xef\xbb\xbf" + b"hello",
            b"\xff\xfe" + b"hello",
            "Sample UTF16".encode("utf-16-be"),
            "Sample UTF16".encode("utf-16-le"),
            b"X" * 4096,
            "Long Unicode data ".encode("utf-16-be") * 220,
        ]
        for data in cases:
            with self.subTest(data=data[:16]):
                self.assertEqual(
                    new_guess(data, known),
                    original_guess(data, known),
                )

    def test_preserves_auxiliary_provenance_fields(self):
        records = [
            (*row, "train", f"source-{i}")
            for i, row in enumerate(rows())
        ]
        selected, dropped = new_subset(records)
        selected_reference, dropped_reference = original_subset(
            [(r[0], r[1], r[2]) for r in records]
        )
        self.assertEqual([r[:3] for r in selected], selected_reference)
        self.assertEqual(dropped, dropped_reference)
        self.assertTrue(all(len(r) == 5 for r in selected))


if __name__ == "__main__":
    unittest.main()
