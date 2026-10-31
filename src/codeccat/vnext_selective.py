"""Experimental conditional specialist: keep a frozen incumbent except Latin ambiguity.

This does NOT implement a fixture-specific override or mutable confidence rule.
The original model decides whether specialized correction is even eligible:
both candidate winners must belong to the Latin/Windows legacy family and
must strictly decode into genuinely DIFFERENT Unicode text.
The family gate protects Unicode, CJK and EBCDIC baseline decisions.
No performance claims and no automatic production promotion.
"""
from __future__ import annotations

import codecs

from .vnext_strong_l2 import Ranked, TrainableDetector
from .vnext_bom import inspect_bom


def _family_eligible(label: str | None) -> bool:
    if label is None:
        return False
    try:
        normalized = codecs.lookup(str(label)).name
    except LookupError:
        return False
    return normalized.startswith("iso8859-") or normalized.startswith("cp12")


def _strict_text(raw: bytes, label: str, *, final: bool) -> str | None:
    try:
        decoder = codecs.getincrementaldecoder(label)("strict")
        return decoder.decode(raw[:4096], final=final and len(raw) <= 4096)
    except (UnicodeError, LookupError, ValueError):
        return None


class SelectiveLatinDetector:
    """Decode-capable adapter that preserves baseline non-Latin decisions.

    Both models are ordinary separately trained portable TrainableDetectors.
    It is intentional that two models must be loaded: accuracy first.
    """

    def __init__(self, baseline: TrainableDetector, specialist: TrainableDetector):
        self.baseline = baseline
        self.specialist = specialist

    @property
    def classifier(self):
        """Supports the existing decode() ASCII-ambiguity contract."""
        return self.baseline.classifier

    @property
    def fingerprint(self) -> str:
        return self.baseline.fingerprint + ":" + self.specialist.fingerprint

    def rank(self, data: bytes, *, final: bool = True, structural_first: bool = False) -> Ranked:
        original = self.baseline.rank(data, final=final, structural_first=structural_first)
        a = original.encoding
        if not _family_eligible(a):
            return original
        if inspect_bom(data, final=final).signature is not None or original.bom_hint:
            return original
        alternative = self.specialist.rank(data, final=final, structural_first=structural_first)
        b = alternative.encoding
        if not _family_eligible(b) or b == a or alternative.bom_hint:
            return original
        # No decision on a corrupted, incompatible or truncated candidate.
        initial_text = _strict_text(data, str(a), final=final)
        proposed_text = _strict_text(data, str(b), final=final)
        if initial_text is None or proposed_text is None or initial_text == proposed_text:
            return original
        # The specialist is a separately trained model; preserve its ranked
        # alternatives and its own uncertainty marker rather than invent
        # combined class scores or claiming calibrated probabilities.
        return alternative
