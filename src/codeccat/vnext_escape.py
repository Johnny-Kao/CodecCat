"""Decisive ISO-2022 escape-designation evidence (no statistical priors).

Only recognized valid designation sequences can select a trained encoding;
partial escape designations defer selection to the ordinary L2 ranking.
"""
from __future__ import annotations
import codecs

JP_DESIGNATIONS=(b"\x1b$B", b"\x1b$@", b"\x1b(B", b"\x1b(J", b"\x1b(I", b"\x1b$(D")
KR_DESIGNATIONS=(b"\x1b$)C",)

def select_escape_encoding(data: bytes, supported: tuple[str,...], *, final: bool) -> str | None:
    """Return a codec only for a non-conflicting recognized designation.

    A designation is evidence, not proof of natural-language plausibility.
    Do not force codecs outside the trained label set.
    """
    prefix=data[:4096]
    # ESC is rare in ordinary text but may be present in terminal output.
    # Require an initial designation, not an arbitrary embedded ESC.
    if not prefix.startswith(b"\x1b"):return None
    jp=any(prefix.startswith(seq) for seq in JP_DESIGNATIONS)
    kr=any(prefix.startswith(seq) for seq in KR_DESIGNATIONS)
    if jp==kr:return None
    desired="iso2022_jp" if jp else "iso2022_kr"
    if desired not in supported:return None
    try:
        decoder=codecs.getincrementaldecoder(desired)("strict")
        decoder.decode(prefix,final=final and len(data)<=4096)
    except (UnicodeError,LookupError):return None
    return desired
