"""Research-only cheap codec-family proposal using 256-bin byte distributions.

A diagnostic candidate generator, NOT a probability model. No decoded language model,
no scanning hundreds of text decodings; strict-decode only top distinct proposals.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import codecs
import numpy as np

def histogram(raw: bytes) -> np.ndarray:
    arr=np.frombuffer(raw[:4096],dtype=np.uint8)
    bins=np.bincount(arr,minlength=256).astype(np.float64)
    # High-byte distributions distinguish 8-bit legacy codepages from UTF8,
    # while low-byte counts capture ASCII-compatible legacy encodings.
    bins/=max(float(arr.size),1.)
    return bins

@dataclass(frozen=True)
class ByteFamilies:
    names: tuple[str,...]
    centroids: np.ndarray

    @classmethod
    def fit(cls,training):
        accum=defaultdict(lambda:np.zeros(256,dtype=np.float64))
        counts=defaultdict(int)
        for raw,label,group in training:
            accum[label]+=histogram(raw)
            counts[label]+=1
        names=tuple(sorted(accum))
        vectors=np.stack([accum[k]/counts[k] for k in names])
        # Normalize using non-ASCII weights to prevent space/newline dominating.
        vectors[:,:128]*=.25
        vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
        return cls(names,vectors)

    def proposals(self,raw:bytes,*,limit:int=8)->tuple[str,...]:
        vec=histogram(raw)
        vec[:128]*=.25
        vec/=max(float(np.linalg.norm(vec)),1e-12)
        order=np.argsort(self.centroids@vec)[::-1]
        return tuple(self.names[int(i)] for i in order[:limit])

def is_valid(raw: bytes,encoding:str)->bool:
    try:
        dec=codecs.getincrementaldecoder(encoding)("strict")
        dec.decode(raw[:4096],final=len(raw)<4096)
        return True
    except (UnicodeError,LookupError,ValueError):
        return False

def candidate_union(raw: bytes, ranked, families:ByteFamilies, limit:int=8)->tuple[str,...]:
    names=list(ranked.candidates)
    # Structural gates are decisive; never override BOM / escaping.
    if ranked.bom_hint or len(ranked.candidates)==1:return tuple(names)
    for label in families.proposals(raw,limit=limit):
        if label not in names and is_valid(raw,label):
            names.append(label)
    return tuple(names)
