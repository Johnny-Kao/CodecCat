"""Strict text decoding for the trainable vNext L1/L2 adapter.

A codec label and BOM handling are separate contracts. This file leaves the
older vnext_pipeline prototype and all published legacy APIs untouched.
"""
from __future__ import annotations

import codecs
from dataclasses import dataclass
from typing import Literal

from .vnext_bom import inspect_bom
from .vnext_strong_l2 import TrainableDetector
from .vnext_l3_language import LanguageCoherence

BomPolicy = Literal["preserve", "strip"]

@dataclass(frozen=True)
class Detection:
    encoding: str | None
    text: str | None
    candidates: tuple[str, ...]
    bom_encoding: str | None
    bom_present: bool
    ambiguous: bool
    reason: str
    bom_policy: BomPolicy
    l3_called: bool = False
    l3_changed: bool = False

    @property
    def accepted(self) -> bool:
        return self.encoding is not None and self.text is not None


def _literal_decode(raw: bytes, predicted: str) -> str:
    """Preserve an initial signature as a literal Unicode U+FEFF."""
    signature = inspect_bom(raw).encoding
    normalized = predicted.replace("_", "-").lower()
    decode_codec = predicted
    if normalized == "utf-8-sig":
        decode_codec = "utf-8"
    elif normalized in ("utf-16", "utf-32") and signature is not None:
        if signature.startswith(normalized + "-"):
            decode_codec = signature
    return codecs.getincrementaldecoder(decode_codec)("strict").decode(raw, final=True)


def decode(
    model: TrainableDetector,
    raw: bytes,
    *,
    bom_policy: BomPolicy = "preserve",
    structural_first: bool = False,
    language_model: LanguageCoherence | None = None,
) -> Detection:
    """Decode one complete bytes object.

    'preserve' returns the literal initial U+FEFF when present;
    'strip' removes exactly one initial U+FEFF only for recognized BOM bytes.
    Both modes use strict decoding and abstain rather than silently replace
    invalid bytes. This is not a streaming decoder.
    """
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    if bom_policy not in ("preserve", "strip"):
        raise ValueError("bom_policy must be 'preserve' or 'strip'")
    evidence = inspect_bom(raw)
    ranked = model.rank(raw, final=True, structural_first=structural_first)
    l3 = language_model.rerank(ranked, raw, final=True) if language_model is not None else None
    chosen = ranked.encoding if l3 is None else l3.encoding
    candidates = ranked.candidates
    if l3 is not None and l3.changed:
        candidates = (chosen,) + tuple(c for c in candidates if c != chosen)
    attrs = {
        "candidates": candidates,
        "l3_called": l3.called if l3 is not None else False,
        "l3_changed": l3.changed if l3 is not None else False,
        "bom_encoding": evidence.encoding,
        "bom_present": evidence.signature is not None,
        "bom_policy": bom_policy,
    }
    if chosen is None:
        return Detection(None, None, ambiguous=True, reason="no_candidate", **attrs)
    try:
        decoded = _literal_decode(raw, str(chosen))
    except (UnicodeError, LookupError):
        return Detection(None, None, ambiguous=True, reason="invalid_selected_codec", **attrs)
    if bom_policy == "strip" and evidence.signature and decoded.startswith("\ufeff"):
        decoded = decoded[1:]
    # Seven-bit text can be textually identical under multiple trained
    # encodings. Report the historical codec label as uncertain without
    # fabricating a posterior probability or silently discarding the text.
    if (raw and evidence.signature is None and len(raw) <= 4096 and
            all(byte in (9, 10, 13) or 32 <= byte <= 126 for byte in raw)):
        if decoded == raw.decode("ascii"):
            for alternative in model.classifier.classes_:
                if str(alternative) == str(chosen):
                    continue
                try:
                    alt = codecs.getincrementaldecoder(str(alternative))("strict")
                    if alt.decode(raw, final=True) == decoded:
                        return Detection(str(chosen), decoded, ambiguous=True,
                                         reason="textually_ambiguous_ascii", **attrs)
                except (UnicodeError, LookupError):
                    pass
    return Detection(str(chosen), decoded, ambiguous=ranked.ambiguous,
                     reason="selected", **attrs)
