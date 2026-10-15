"""Conservative no-BOM UTF-32 ASCII-range byte-lane evidence."""
from __future__ import annotations
import codecs

def select_utf32_no_bom(data: bytes, supported: tuple[str, ...], *, final: bool) -> str | None:
    prefix=data[:4096]
    if len(prefix)<64:return None
    n=len(prefix)//4
    if n<16:return None
    part=prefix[:n*4]
    zero=[part[j::4].count(0) for j in range(4)]
    # All but one lane must be nearly entirely NUL for ASCII-range UTF-32,
    # and the primary lane must be visibly non-NUL.
    best=None
    for codec,primary in (("utf-32-le",0),("utf-32-be",3)):
        if codec not in supported:continue
        if zero[primary]>n//4:continue
        if any(z < (n*9)//10 for i,z in enumerate(zero) if i!=primary):continue
        try:
            dec=codecs.getincrementaldecoder(codec)("strict")
            dec.decode(prefix,final=final and len(data)<=4096)
        except (UnicodeError,LookupError):continue
        if best is not None:return None
        best=codec
    return best
