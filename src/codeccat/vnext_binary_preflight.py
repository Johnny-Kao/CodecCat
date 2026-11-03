"""Research-only conservative binary/text abstention preflight.

Not a general MIME sniffer. Requires independent semantic acceptance validation
before integration; never infers text from decodability alone.
"""
from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
from .vnext_bom import inspect_bom
import codecs

MAGIC={
    "png":bytes.fromhex("89504e470d0a1a0a"),
    "jpeg":bytes.fromhex("ffd8ff"),
    "gzip":bytes.fromhex("1f8b08"),
    "zip":bytes.fromhex("504b0304"),
    "elf":bytes.fromhex("7f454c46"),
    "wasm":bytes.fromhex("0061736d01000000"),
}
@dataclass(frozen=True)
class TextPreflight:
    text_possible: bool
    reason: str
    confidently_binary: bool

def _unicode_nul_shape(raw:bytes)->bool:
    b=raw[:4096]
    if len(b)<32:return False
    if len(b)%2:
        b=b[:-1]
    n=len(b)//2
    even=b[::2].count(0)
    odd=b[1::2].count(0)
    dominant=max(even,odd)
    return dominant>=max(12,int(.30*n)) and dominant>=4*max(1,min(even,odd))

def _utf32_shape(b:bytes)->bool:
    """Validate zero high-byte alignment AND actual Unicode text, not mere decode."""
    prefix=b[:4096]
    groups=len(prefix)//4
    if groups<16:return False
    sample=prefix[:groups*4]
    for encoding,hi,next_hi in (("utf-32-be",0,1),("utf-32-le",3,2)):
        high=sample[hi::4]
        second=sample[next_hi::4]
        if high.count(0)<int(groups*.85):
            continue
        if sum(x<=0x10 for x in second)<int(groups*.90):
            continue
        try:
            text=sample.decode(encoding,"strict")
        except UnicodeError:
            continue
        if sum(ch.isprintable() or ch in "\\r\\n\\t" for ch in text)>=int(len(text)*.90):
            return True
    return False

def inspect_text(raw:bytes)->TextPreflight:
    if not isinstance(raw,bytes):raise TypeError("raw must be bytes")
    if not raw:return TextPreflight(True,"empty",False)
    prefix=raw[:4096]
    bom=inspect_bom(prefix)
    if bom.signature:return TextPreflight(True,"bom",False)
    for name,signature in MAGIC.items():
        if prefix.startswith(signature):
            return TextPreflight(False,"binary-magic-"+name,True)
    if len(prefix)<64:return TextPreflight(True,"short",False)
    n=len(prefix)
    if prefix.count(0)==n:
        return TextPreflight(False,"all-null",True)
    if len(set(prefix))<=2 and prefix.count(0)>=n//3:
        return TextPreflight(False,"repetitive-null",True)
    # Do not reject UTF16/UTF32 based solely on null/control frequency.
    if _unicode_nul_shape(prefix):
        return TextPreflight(True,"unicode-null-structure",False)
    if _utf32_shape(prefix):
        return TextPreflight(True,"utf32-byte-structure",False)
    if b"\\x1b$" in prefix or b"\\x1b(" in prefix:
        return TextPreflight(True,"escape-structure",False)
    # A stateful escape encoding can contain legitimate control-like bytes.
    # Require a strict decode producing predominantly printable Unicode text.
    for name in ("iso2022_kr", "iso2022_jp", "iso2022_jp_2"):
        try:
            text=prefix.decode(name,"strict")
        except (UnicodeError,LookupError):
            continue
        if len(text)>=16 and sum(ch.isprintable() or ch in "\\r\\n\\t" for ch in text)>=int(len(text)*.90):
            return TextPreflight(True,"stateful-text-structure",False)
    controls=sum((b<32 and b not in (9,10,13)) or b==127 for b in prefix)
    if controls/n>=.20 and prefix.count(10)<=1:
        return TextPreflight(False,"control-density",True)
    if n>=256:
        freq=Counter(prefix)
        entropy=-sum((v/n)*math.log2(v/n) for v in freq.values())
        if entropy>7.3:
            try:prefix.decode("utf-8")
            except UnicodeDecodeError:
                return TextPreflight(False,"high-entropy-nonutf8",True)
    return TextPreflight(True,"unresolved",False)
