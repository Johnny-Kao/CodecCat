"""Encoding-conditional byte bigram evidence; experimental conservative reranker.

Trained only on permitted group-heldout rows. Score candidate decoded codecs
with a per-codec byte bigram log-likelihood ratio against corpus background.
Only top-3 legal candidates and low-margin L2 decisions are eligible.
No test labels, source metadata or competitor packages at inference time.
"""
from __future__ import annotations
import math
from collections import Counter,defaultdict
from dataclasses import dataclass
import numpy as np
from codeccat.vnext_l3_language import Rerank

MAX_MARGIN=0.50
MIN_GAIN=0.25
MIN_SAMPLES=3
MIN_PAIRS=512
PREFIX=4096
SMOOTH=2048.0

def pairs(data):
    a=np.frombuffer(data[:PREFIX],dtype=np.uint8)
    if len(a)<2:return Counter()
    keys=(a[:-1].astype(np.int32)*256+a[1:].astype(np.int32))
    values, counts=np.unique(keys,return_counts=True)
    return dict(zip(values.tolist(),counts.tolist()))

class ConditionalByteBigram:
    def __init__(self,bg,bg_total,classes,totals,doc_counts):
        self.bg=bg;self.bg_total=bg_total
        self.classes=classes;self.totals=totals;self.doc_counts=doc_counts

    @classmethod
    def fit(cls,rows):
        bg=Counter();class_counts=defaultdict(Counter)
        total=Counter();doc_count=Counter()
        for raw,codec,group in rows:
            stats=pairs(raw)
            if not stats:continue
            bg.update(stats);class_counts[codec].update(stats)
            total[codec]+=sum(stats.values());doc_count[codec]+=1
        bg_total=sum(bg.values())
        return cls(bg,bg_total,dict(class_counts),dict(total),dict(doc_count))

    def score(self,raw,codec):
        c=self.classes.get(codec)
        n=self.totals.get(codec,0)
        if c is None or self.doc_counts.get(codec,0)<MIN_SAMPLES or n<MIN_PAIRS:return None
        query=pairs(raw)
        if not query:return None
        total=sum(query.values());value=0.0
        for pair,frequency in query.items():
            background=(self.bg.get(pair,0)+1)/(self.bg_total+65536)
            conditional=(c.get(pair,0)+SMOOTH*background)/(n+SMOOTH)
            value+=frequency*math.log(conditional/background)
        return value/total

    def rerank(self,ranked,raw,*,final=True,max_l2_margin=.5,min_language_advantage=.15):
        first=ranked.encoding
        if first is None or len(ranked.candidates)<2 or len(ranked.scores)<2:
            return Rerank(first,False,False,"insufficient-candidates")
        margin=ranked.scores[0]-ranked.scores[1]
        if not(0<=margin<=MAX_MARGIN):
            return Rerank(first,False,False,"byte-margin-decisive")
        scores={enc:self.score(raw,enc) for enc in ranked.candidates[:3]}
        baseline=scores[first]
        if baseline is None:return Rerank(first,True,False,"no-trained-baseline-statistics")
        # Only rerank among already strictly validated candidates.
        alternatives=[(score,enc) for enc,score in scores.items()
                      if enc!=first and score is not None]
        if not alternatives:return Rerank(first,True,False,"no-trained-alternative")
        best_value,best_enc=max(alternatives)
        if best_value-baseline<=MIN_GAIN:
            return Rerank(first,True,False,"no-strong-conditional-advantage",best_value-baseline)
        return Rerank(best_enc,True,True,"conditional-byte-bigram",best_value-baseline)
