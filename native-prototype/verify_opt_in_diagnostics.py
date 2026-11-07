"""Contract tests for experimental, opt-in, non-destructive diagnostics.

Run against an installed staging Wheel: no training dependencies required.
"""
from __future__ import annotations

from dataclasses import asdict, FrozenInstanceError
import importlib.util
import time

import codeccat
from codeccat.diagnostics import AmbiguityDiagnostic, inspect_ambiguity
from codeccat.vnext_decoding import Detection


def fake(raw: bytes, selected: str, candidates: tuple[str, ...],
         *, bom: bool = False, policy: str = "strip") -> Detection:
    return Detection(
        encoding=selected, text=raw.decode(selected, "strict"),
        candidates=candidates, bom_encoding="utf-8-sig" if bom else None,
        bom_present=bom, ambiguous=False, reason="selected",
        bom_policy=policy, l3_called=False, l3_changed=False,
    )


def run():
    # Legitimate Latin-1 U+0080 and CP1252 Euro are both valid interpretations.
    raw = b"Price: \x80; total."
    selected = fake(raw, "iso-8859-1", ("iso-8859-1", "cp1252"))
    snapshot = asdict(selected)
    diag = inspect_ambiguity(raw, selected)
    assert diag == AmbiguityDiagnostic(
        True, "c1_interpretation_ambiguous", ("cp1252",)
    ), diag
    assert asdict(selected) == snapshot
    assert selected.text == "Price: \x80; total."
    assert codeccat.decode(b"hello") == "hello"

    # Buffer input accepted with identical semantics.
    assert inspect_ambiguity(bytearray(raw), selected) == diag
    assert inspect_ambiguity(memoryview(raw), selected) == diag
    assert not inspect_ambiguity(b"Different data", selected).warning

    # A legitimate C1 should be exposed as uncertainty rather than "fixed".
    assert "\x80" in selected.text
    assert not selected.ambiguous
    assert diag.warning  # original 'ambiguous' field deliberately unchanged

    # BOM is not challenged, including when a test-specific result has C1.
    with_bom = fake(raw, "iso-8859-1", ("iso-8859-1", "cp1252"), bom=True)
    assert inspect_ambiguity(raw, with_bom).reason == "bom_present"

    # Existing model L3 and BOM metadata remain untouched, too.
    from dataclasses import replace
    l3 = replace(selected, l3_called=True, l3_changed=True)
    assert inspect_ambiguity(raw, l3).warning
    assert l3.l3_called and l3.l3_changed and l3.text == selected.text

    # No alternate strict decoder with materially different C1-free text.
    no_alt = fake(raw, "iso-8859-1", ("iso-8859-1", "ascii"))
    assert not inspect_ambiguity(raw, no_alt).warning

    # Duplicated aliases cannot produce duplicate alternative reports.
    aliases = fake(raw, "latin-1", ("latin1", "cp1252", "windows-1252"))
    assert inspect_ambiguity(raw, aliases).alternative_encodings == ("cp1252",)

    # Ordinary UTF-8/CJK has a cheap short-circuit with no candidate decoding.
    for text in ("Hello World!" * 50, "日本語の内容" * 50, "café" * 50):
        encoded = text.encode("utf-8")
        detected = codeccat.detect(encoded)
        before = asdict(detected)
        report = inspect_ambiguity(encoded, detected)
        assert report.warning is False, report
        assert asdict(detected) == before

    no_result = Detection(
        encoding=None, text=None, candidates=(), bom_encoding=None,
        bom_present=False, ambiguous=True, reason="no_candidate",
        bom_policy="strip",
    )
    assert inspect_ambiguity(b"\xff", no_result).reason == "no_decoded_text"

    for data, result in (("not-bytes", selected), (raw, "not-detection")):
        try:
            inspect_ambiguity(data, result)
        except TypeError:
            pass
        else:
            raise AssertionError("expected input type validation")

    assert importlib.util.find_spec("numpy") is None
    assert importlib.util.find_spec("sklearn") is None
    print("OPT_IN_C1_DIAGNOSTICS_CONTRACT_PASS")


if __name__ == "__main__":
    run()
