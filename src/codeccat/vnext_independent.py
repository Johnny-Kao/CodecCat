"""Independent vNext baseline: trained byte-transition language likelihood.

This module intentionally does not import Runtime, R21, HMT768 or R21 specialists.
A fixed structural candidate validator precedes configurable statistical scoring.
Research prototype: fit from supplied labelled bytes; *not* a production detector.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from math import log
import numpy as np
from .logic_first import _validate

SCHEMA="independent-byte-transitions-v1"
DIM=2048

def vector(data: bytes) -> np.ndarray:
    """Unordered sampling is forbidden: preserves directional adjacent transitions."""
    sample=data[:4096]
    a=np.frombuffer(sample,dtype=np.uint8)
    bins=np.zeros(DIM,dtype=np.float64)
    if len(a)>1:
        x=a[:-1].astype(np.uint32)
        y=a[1:].astype(np.uint32)
        idx=((x*257+y)*2654435761 & 0xFFFFFFFF) % DIM
        bins=np.bincount(idx.astype(np.int64),minlength=DIM).astype(np.float64)
    return bins

@dataclass(frozen=True)
class TransitionModel:
    labels: tuple[str,...]
    log_probs: np.ndarray
    log_priors: np.ndarray
    schema: str=SCHEMA

    def rank(self,data: bytes) -> tuple[str,...]:
        if self.schema!=SCHEMA:raise ValueError("incompatible schema")
        v=vector(data)
        raw=self.log_probs@v+self.log_priors
        # Structural invalidation is *candidate-specific*, and does not
        # penalize unknown codec names or a partial last code unit.
        scores=[]
        for i,label in enumerate(self.labels):
            validity=_validate(data[:4096],label).status if label in ("utf-8","shift-jis","euc-jp","big5","gb18030") else "unknown"
            scores.append((float(raw[i]) if validity!="invalid" else float("-inf"),label))
        if all(score==float("-inf") for score,_ in scores):
            scores=[(float(raw[i]),label) for i,label in enumerate(self.labels)]
        return tuple(label for _,label in sorted(scores,key=lambda item:(-item[0],item[1])))

def fit(samples, smoothing:float=0.5):
    """Fit once from any iterable of (raw_bytes, normalized_encoding) pairs."""
    counts=defaultdict(lambda:np.zeros(DIM,dtype=np.float64))
    freq=defaultdict(int)
    for data,label in samples:
        if not isinstance(data,bytes) or not isinstance(label,str) or not label:
            raise TypeError("expected (bytes, nonempty canonical encoding)")
        counts[label]+=vector(data)
        freq[label]+=1
    labels=tuple(sorted(counts))
    if len(labels)<2:raise ValueError("need at least two classes")
    lp=[]
    priors=[]
    total=sum(freq.values())
    for label in labels:
        c=counts[label]+smoothing
        lp.append(np.log(c/c.sum()))
        priors.append(log(freq[label]/total))
    return TransitionModel(labels,np.stack(lp),np.asarray(priors))
