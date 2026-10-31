"""CodecCat vNext stable structural gate, independent of old detector.

Only proven impossibility removes a candidate. BOM is evidence, never a
forced label. Partial inputs preserve incomplete candidates.
"""
from __future__ import annotations
import codecs
from dataclasses import dataclass
from .vnext_bom import inspect_bom, BomEvidence

@dataclass(frozen=True)
class GateResult:
    possible: tuple[str, ...]
    impossible: tuple[str, ...]
    incomplete: tuple[str, ...]
    bom: BomEvidence

def evaluate(data: bytes, candidates: tuple[str, ...], *, final: bool = True) -> GateResult:
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    possible, impossible, incomplete = [], [], []
    for name in dict.fromkeys(candidates):
        try:
            dec = codecs.getincrementaldecoder(name)(errors="strict")
        except LookupError:
            # Unknown codecs are not proven invalid: preserve and let the
            # caller decide whether unsupported codecs are eligible.
            incomplete.append(name)
            possible.append(name)
            continue
        try:
            dec.decode(data, final=final)
            if not final and dec.getstate()[0]:
                incomplete.append(name)
            possible.append(name)
        except UnicodeError as exc:
            if not isinstance(exc, UnicodeDecodeError):
                impossible.append(name)
                continue
            if not final and exc.end == len(data) and (
                "unexpected end" in exc.reason.lower()
                or "incomplete" in exc.reason.lower()
                or "truncated" in exc.reason.lower()
            ):
                incomplete.append(name)
                possible.append(name)
            else:
                impossible.append(name)
    return GateResult(tuple(possible), tuple(impossible), tuple(incomplete), inspect_bom(data, final=final))
