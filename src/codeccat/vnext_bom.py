"""Stable BOM parsing; longest signature first.

Returns evidence only, never forces unsupported model label.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BomEvidence:
    encoding: str | None
    signature: bytes | None
    incomplete: bool

_SIGNATURES = (
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xfe\xff", "utf-16-be"),
    (b"\xff\xfe", "utf-16-le"),
)

def inspect_bom(data: bytes, final: bool = True) -> BomEvidence:
    if not isinstance(data, bytes):
        raise TypeError("expected bytes")
    if not final and data and any(sig.startswith(data) and len(data) < len(sig) for sig, _ in _SIGNATURES):
        return BomEvidence(None, None, True)
    for sig, encoding in _SIGNATURES:
        if data.startswith(sig):
            return BomEvidence(encoding, sig, False)
    return BomEvidence(None, None, False)
