"""CodecCat vNext isolated trainable L1→L2→selective L3 pipeline.

No legacy Runtime or R21 model imports. Standard-library-only engine.
Model is a development prototype, not an approved replacement.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from .vnext_gate import evaluate

SCHEMA = "vnext-pipeline-1"

def transitions(data: bytes) -> Counter[int]:
    seq = data[:4096]
    return Counter((seq[i] << 8) | seq[i + 1] for i in range(len(seq)-1))

def train(samples: list[tuple[bytes,str,str]], *, alpha: float = 0.5) -> "Model":
    if not samples: raise ValueError("empty training set")
    if alpha <= 0: raise ValueError("alpha must be positive")
    labels: dict[str,Counter[int]] = defaultdict(Counter)
    counts = Counter()
    group_labels: dict[str,set[str]] = defaultdict(set)
    seen_bytes: dict[str,str] = {}
    digest = hashlib.sha256()
    for data,label,group in samples:
        if not isinstance(data,bytes) or not label or not group:
            raise ValueError("expected bytes, nonempty encoding and source_group")
        key = hashlib.sha256(data).hexdigest()
        if key in seen_bytes and seen_bytes[key] != label:
            raise ValueError("identical bytes have conflicting labels")
        seen_bytes[key] = label
        group_labels[group].add(label)
        labels[label].update(transitions(data))
        counts[label] += 1
        digest.update(len(data).to_bytes(8,"big"))
        digest.update(data)
        digest.update(label.encode())
        digest.update(group.encode())
    return Model(
        labels=tuple(sorted(labels)),
        counts=dict(counts),
        evidence={label:dict(c) for label,c in labels.items()},
        alpha=alpha,
        fingerprint=digest.hexdigest(),
    )

@dataclass(frozen=True)
class Result:
    encoding: str | None
    confidence: float
    candidates: tuple[str,...]
    l3_called: bool
    ambiguous: bool
    bom_hint: str | None

@dataclass(frozen=True)
class Model:
    labels: tuple[str,...]
    counts: dict[str,int]
    evidence: dict[str,dict[int,int]]
    alpha: float
    fingerprint: str
    schema: str=SCHEMA

    def save(self,path: str|Path) -> None:
        data={
          "schema":self.schema,"labels":list(self.labels),"counts":self.counts,
          "evidence":{k:{str(a):b for a,b in v.items()} for k,v in self.evidence.items()},
          "alpha":self.alpha,"fingerprint":self.fingerprint
        }
        Path(path).write_text(json.dumps(data,sort_keys=True,separators=(",",":")))

    @classmethod
    def load(cls,path: str|Path) -> "Model":
        d=json.loads(Path(path).read_text())
        if d.get("schema")!=SCHEMA:raise ValueError("incompatible schema")
        return cls(tuple(d["labels"]),{k:int(v) for k,v in d["counts"].items()},
                   {k:{int(a):int(b) for a,b in z.items()} for k,z in d["evidence"].items()},
                   float(d["alpha"]),d["fingerprint"])

    def detect(self,data:bytes,*,final:bool=True,margin_threshold:float=0.2) -> Result:
        if not isinstance(data,bytes):raise TypeError("data must be bytes")
        if not self.labels:raise ValueError("no classes")
        gate=evaluate(data[:4096],self.labels,final=final and len(data)<=4096)
        viable=gate.possible
        if not viable:
            return Result(None,0.0,(),False,True,gate.bom.encoding)
        if len(viable)==1:
            return Result(viable[0],1.0,viable,False,False,gate.bom.encoding)
        obs=transitions(data)
        # Constant-time BOM evidence may rank supported compatible labels, but
        # an untrained BOM codec is never silently substituted by a different one.
        bom_codec=gate.bom.encoding
        if bom_codec:
            family=bom_codec.split("-")[:2]
            supported_bom=[c for c in viable if c.startswith("-".join(family))]
            if supported_bom and len(supported_bom)==1:
                return Result(supported_bom[0],1.0,tuple(supported_bom),False,False,bom_codec)
        n=max(1,len(data[:4096])-1)
        total_examples=sum(self.counts.values())
        scores={}
        for label in viable:
            evidence=self.evidence[label]
            denominator=sum(evidence.values()) + self.alpha*65536
            # normalized cross-entropy per adjacent pair, length-independent
            ll=sum(count*math.log((evidence.get(key,0)+self.alpha)/denominator)
                   for key,count in obs.items())/n
            scores[label]=ll+0.05*math.log(self.counts[label]/total_examples)
        ordered=sorted(scores,key=lambda c:(-scores[c],c))
        if len(ordered)<2:
            return Result(ordered[0],1.0,tuple(ordered),False,False,gate.bom.encoding)
        gap=scores[ordered[0]]-scores[ordered[1]]
        # L3 has to use a different signal: decoded Unicode printability and
        # control characters on the top two, never repeat L1 validity.
        need_l3=gap<margin_threshold and len(data)>=16
        effective_scores=dict(scores)
        if need_l3:
            revised={}
            for label in ordered[:2]:
                try:txt=data[:4096].decode(label,"strict")
                except (UnicodeDecodeError,LookupError):continue
                if not txt:continue
                controls=sum((ord(ch)<32 and ch not in "\t\n\r") or 0x7f<=ord(ch)<=0x9f for ch in txt)
                revised[label]=scores[label]-0.2*controls/len(txt)
            if revised:
                effective_scores.update(revised)
                ordered[:2]=sorted(ordered[:2],key=lambda x:(-effective_scores[x],x))
        best=ordered[0]
        final_gap=effective_scores[ordered[0]]-effective_scores[ordered[1]]
        confidence=1/(1+math.exp(-min(max(final_gap,0.0),80)))
        uncertain=final_gap<0.015 and not gate.bom.encoding
        return Result(None if uncertain else best,confidence,tuple(ordered),need_l3,uncertain,gate.bom.encoding)
