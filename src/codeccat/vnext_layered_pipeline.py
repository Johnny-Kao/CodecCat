"""Opt-in research pipeline: semantic abstain -> deterministic evidence -> L2 -> L3.

No default runtime / released alpha changes. This module defines the integrated
decision contract for future promotion, not a public stable API.
"""
from __future__ import annotations
from dataclasses import dataclass
from .vnext_binary_preflight import inspect_text
from .vnext_declarations import choose
from .vnext_decoding import _literal_decode
from .vnext_bom import inspect_bom

@dataclass(frozen=True)
class LayeredDetection:
    encoding: str | None
    text: str | None
    accepted: bool
    reason: str
    candidates: tuple[str,...]
    declaration_used: bool
    language_changed: bool

def detect(model, raw:bytes, *, language_model=None, reject_binary:bool=True, bom_policy:str="strip")->LayeredDetection:
    if not isinstance(raw,bytes):raise TypeError("raw must be bytes")
    if bom_policy not in ("strip","preserve"):raise ValueError("invalid BOM policy")
    if reject_binary:
        gate=inspect_text(raw)
        if not gate.text_possible:
            return LayeredDetection(None,None,False,gate.reason,tuple(),False,False)
    ranked=model.rank(raw,final=True)
    if ranked.encoding is None:
        return LayeredDetection(None,None,False,"no-candidate",ranked.candidates,False,False)
    declaration=choose(raw,ranked.encoding)
    declared=declaration.reason.startswith("declared-")
    changed=False
    if declared:
        encoding=declaration.encoding
        reason=declaration.reason
    elif language_model is not None:
        l3=language_model.rerank(ranked,raw,final=True,max_l2_margin=.5,min_language_advantage=.15)
        encoding=l3.encoding
        changed=l3.changed
        reason="language-evidence" if changed else "model-ranked"
    else:
        encoding=ranked.encoding
        reason="model-ranked"
    try:
        text=_literal_decode(raw,str(encoding))
    except (UnicodeError,LookupError,ValueError):
        return LayeredDetection(None,None,False,"invalid-selected-codec",ranked.candidates,declared,changed)
    # UTF-7 represents its optional leading signature as shifted U+FEFF.
    # Keep signature recognition separate from the L1 codec selection: the
    # bytes are ASCII-compatible and are only a BOM when UTF-7 was selected.
    utf7_signature = (str(encoding).replace("_", "-").lower() == "utf-7"
                      and raw.startswith((b"+/v8", b"+/v9", b"+/v+", b"+/v/")))
    if bom_policy=="strip" and (inspect_bom(raw).signature or utf7_signature) and text.startswith("\ufeff"):
        text=text[1:]
    return LayeredDetection(str(encoding),text,True,reason,ranked.candidates,declared,changed)
