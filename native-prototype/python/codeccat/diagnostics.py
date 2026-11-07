"""Experimental, opt-in evidence of ambiguous C1-control interpretation.

This module NEVER overrides the chosen encoding, decoded text, candidates,
or ambiguity flag. A warning is not proof of misclassification.
No new model and no optional dependencies are needed.
"""
from __future__ import annotations

import codecs
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vnext_decoding import Detection


@dataclass(frozen=True)
class AmbiguityDiagnostic:
    """Supplemental evidence; not a calibrated confidence or new detection."""

    warning: bool
    reason: str
    alternative_encodings: tuple[str, ...]


def _c1(text: str) -> bool:
    return any("\u0080" <= char <= "\u009f" for char in text)


def inspect_ambiguity(
    data: bytes | bytearray | memoryview,
    result: Detection,
) -> AmbiguityDiagnostic:
    """Inspect one completed detection without mutating it.

    If the selected text contains C1 controls, look only at already-ranked
    alternative codecs. Warn when a strict alternative produces materially
    different text with no C1 controls. Genuine C1 characters may be intended.
    A warning is NEVER a recommendation to substitute another decoding.

    A BOM is treated as decisive evidence and not challenged by this test.
    The same original bytes used for result must be supplied.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    from .vnext_decoding import Detection
    if not isinstance(result, Detection):
        raise TypeError("result must be a codeccat Detection")
    raw = bytes(data)
    if result.text is None or result.encoding is None:
        return AmbiguityDiagnostic(False, "no_decoded_text", ())
    if result.bom_present:
        return AmbiguityDiagnostic(False, "bom_present", ())
    if not _c1(result.text):
        return AmbiguityDiagnostic(False, "no_c1_controls", ())

    # A mismatched raw/result pair should never produce false evidence.
    try:
        selected = raw.decode(result.encoding, "strict")
    except (UnicodeError, LookupError):
        return AmbiguityDiagnostic(False, "selected_codec_not_reproducible", ())
    if selected != result.text:
        return AmbiguityDiagnostic(False, "result_not_reproducible", ())

    alternatives: list[str] = []
    try:
        selected_codec = codecs.lookup(result.encoding).name
    except LookupError:
        return AmbiguityDiagnostic(False, "selected_codec_not_reproducible", ())
    seen = {selected_codec}
    for label in result.candidates:
        try:
            canonical = codecs.lookup(label).name
        except (LookupError, TypeError):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        try:
            other_text = raw.decode(canonical, "strict")
        except (LookupError, UnicodeError):
            continue
        if other_text != result.text and not _c1(other_text):
            alternatives.append(label)
    if alternatives:
        return AmbiguityDiagnostic(True, "c1_interpretation_ambiguous", tuple(alternatives))
    return AmbiguityDiagnostic(False, "no_alternative_without_c1", ())
