"""Explicit document charset evidence, research-only deterministic selector.

Only declaration recognized in the initial 1024 bytes. Does not trust arbitrary
keyword occurrences or override Unicode BOMs. No external metadata involved.
"""
from __future__ import annotations
import codecs,re
from dataclasses import dataclass
from .vnext_bom import inspect_bom
_XML=re.compile(br'^\s*<\?xml\s+[^>]{0,256}?encoding\s*=\s*["\']([A-Za-z0-9._-]{2,40})["\']',re.I)
_HTML=re.compile(br'<meta\s+[^>]{0,512}?charset\s*=\s*["\']?([A-Za-z0-9._-]{2,40})',re.I)
_CSS=re.compile(br'^\s*@charset\s+["\']([A-Za-z0-9._-]{2,40})["\']\s*;',re.I)
_PY=re.compile(br'coding\s*[:=]\s*([-\w.]{2,40})',re.I)
@dataclass(frozen=True)
class Declared:
    encoding: str | None
    reason: str

def choose(raw: bytes,baseline:str|None)->Declared:
    if not isinstance(raw,bytes):raise TypeError("raw must be bytes")
    if inspect_bom(raw).encoding:
        return Declared(baseline,"bom-priority")
    header=raw[:1024]
    found=None;kind=None
    for name,regex in (("xml",_XML),("css",_CSS),("html",_HTML)):
        match=regex.search(header)
        if match:
            found=match.group(1);kind=name;break
    if found is None:
        first_two=b"\n".join(header.splitlines()[:2])
        if (b"coding" in first_two.lower() and
            (first_two.lstrip().startswith(b"#") or b"\n#" in first_two)):
            match=_PY.search(first_two)
            if match:found=match.group(1);kind="python"
    if found is None:return Declared(baseline,"no-declaration")
    try:
        label=found.decode("ascii")
        canonical=codecs.lookup(label).name
        raw.decode(canonical,"strict")
    except (UnicodeError,LookupError,ValueError):
        return Declared(baseline,"invalid-declaration")
    return Declared(canonical,"declared-"+kind)
