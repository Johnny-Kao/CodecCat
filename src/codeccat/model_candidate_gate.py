"""Fail-closed model candidate validation; research-only, no publishing side effects."""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable
import json

@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: tuple[str,...]
    report: dict

def _decoded(data:bytes, label:str|None, final:bool):
    if label is None:return None
    import codecs
    try:
        decoder=codecs.getincrementaldecoder(str(label))("strict")
        return decoder.decode(data[:4096],final=final)
    except (UnicodeError,LookupError,ValueError):
        return None

def validate_candidate(incumbent,candidate,heldout:Iterable[tuple[bytes,str,str]],
                       *,min_samples:int=20,max_net_loss:int=0,
                       max_family_loss:int=0)->GateDecision:
    """No truth-label leakage into prediction; all data must be independently held out.

    Cannot certify provenance/isolation by itself. A fail is always a no-promotion.
    """
    rows=list(heldout)
    reasons=[]
    if min_samples<1 or max_net_loss<0 or max_family_loss<0:raise ValueError("invalid policy")
    if len(rows)<min_samples:reasons.append("insufficient-heldout")
    if not rows: return GateDecision(False,tuple(reasons or ["empty-heldout"]),{"n":0})
    if not all(isinstance(d,bytes) and isinstance(y,str) and y and isinstance(g,str) and g for d,y,g in rows):
        raise ValueError("invalid heldout row")
    if len({sha256(d).hexdigest() for d,_,_ in rows})!=len(rows):
        reasons.append("duplicate-heldout-bytes")
    by_family=defaultdict(lambda:Counter())
    base_correct=new_correct=0
    failures=[]
    for raw,label,group in rows:
        final=len(raw)<4096
        truth=_decoded(raw,label,final)
        if truth is None:
            reasons.append("invalid-reference-decode")
            break
        old=incumbent.rank(raw,final=final).encoding
        new=candidate.rank(raw,final=final).encoding
        old_text=_decoded(raw,old,final)
        new_text=_decoded(raw,new,final)
        was=old_text is not None and old_text==truth
        now=new_text is not None and new_text==truth
        base_correct+=int(was);new_correct+=int(now)
        fam=label.split("-")[0]
        counts=by_family[fam];counts["n"]+=1;counts["old"]+=int(was);counts["new"]+=int(now)
        if was and not now: failures.append({"sha256":sha256(raw).hexdigest(),"family":fam,"source_group":group})
    if new_correct-base_correct < -max_net_loss:reasons.append("aggregate-regression")
    if any(v["new"]-v["old"] < -max_family_loss for v in by_family.values()):
        reasons.append("family-regression")
    report={"schema":"codeccat-model-gate-v1","n":len(rows),"old_correct":base_correct,
            "candidate_correct":new_correct,"regressions":len(failures),
            "families":{k:dict(v) for k,v in sorted(by_family.items())},
            "regressed_sha256":failures,
            "incumbent_fingerprint":getattr(incumbent,"fingerprint",None),
            "candidate_fingerprint":getattr(candidate,"fingerprint",None)}
    return GateDecision(not reasons,tuple(sorted(set(reasons))),report)
