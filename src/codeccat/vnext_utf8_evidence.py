"""Optional evidence-based UTF-8 structural rescue (research candidate).

The UTF-8 codec is characterized by exact byte validity; matching byte
sequences can nevertheless have other historical encoding interpretations.
Therefore a valid Unicode decoding is NOT treated as proof of origin.

This wrapper does not train a separate model or alter its weights. It only
supplies evidence for a supported UTF-8 candidate when the learned model
misses it despite substantial, fully valid non-ASCII UTF-8 content.
"""
from __future__ import annotations

import unicodedata

from .vnext_bom import inspect_bom
from .vnext_strong_l2 import Ranked

MIN_NONASCII_CODEPOINTS=16
MIN_BYTES=64


def safe_utf8_evidence(raw: bytes, *, min_codepoints: int = MIN_NONASCII_CODEPOINTS) -> bool:
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    if min_codepoints<1:
        raise ValueError("min_codepoints must be positive")
    if len(raw)<MIN_BYTES:
        return False
    if inspect_bom(raw, final=True).signature:
        return False
    # rank() uses up to 4096 bytes. Never infer complete UTF-8 from an
    # incomplete multibyte character at the end of that prefix.
    prefix=raw[:4096]
    try:
        text=prefix.decode("utf-8","strict")
    except UnicodeDecodeError:
        return False
    if sum(ord(c)>127 for c in text)<min_codepoints:
        return False
    if any(unicodedata.category(c)=="Cc" and c not in ("\t","\r","\n") for c in text):
        return False
    return True


class Utf8EvidenceDetector:
    """Optional wrapper around a trainable detector or the Latin specialist.

    Preserve deterministic BOM/escape/no-candidate behavior and every existing
    prediction except the evidenced non-UTF-8 winner. An altered label remains
    ambiguous; 1.0 is a structural rank placeholder, not a posterior.
    """
    def __init__(self, baseline, *, min_codepoints: int = MIN_NONASCII_CODEPOINTS):
        if min_codepoints<1:
            raise ValueError("min_codepoints must be positive")
        self.baseline=baseline
        self.min_codepoints=min_codepoints

    @property
    def classifier(self):
        return self.baseline.classifier

    @property
    def fingerprint(self):
        return self.baseline.fingerprint+f":utf8-evidence-v1:{self.min_codepoints}"

    def rank(self, raw: bytes, *, final: bool = True, structural_first: bool = False) -> Ranked:
        original=self.baseline.rank(raw,final=final,structural_first=structural_first)
        if not final or original.encoding is None or original.bom_hint:
            return original
        if str(original.encoding).replace("_","-") in ("utf-8","utf-8-sig"):
            return original
        supported=any(str(codec).replace("_","-")=="utf-8" for codec in self.classifier.classes_)
        if not supported:
            return original
        if not safe_utf8_evidence(raw,min_codepoints=self.min_codepoints):
            return original
        # Structural evidence is NOT proof that the historic codec was UTF-8.
        # Do not claim a calibrated score or fabricate a full candidate list.
        return Ranked("utf-8",("utf-8",),(1.0,),None,True)
