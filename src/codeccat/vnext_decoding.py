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
    attrs = {
        "candidates": ranked.candidates,
        "bom_encoding": evidence.encoding,
        "bom_present": evidence.signature is not None,
        "bom_policy": bom_policy,
    }
    if ranked.encoding is None:
        return Detection(None, None, ambiguous=True, reason="no_candidate", **attrs)
    try:
        decoded = _literal_decode(raw, str(ranked.encoding))
    except (UnicodeError, LookupError):
        return Detection(None, None, ambiguous=True, reason="invalid_selected_codec", **attrs)
    if bom_policy == "strip" and evidence.signature and decoded.startswith("\ufeff"):
        decoded = decoded[1:]
    return Detection(str(ranked.encoding), decoded, ambiguous=ranked.ambiguous,
                     reason="selected", **attrs)
