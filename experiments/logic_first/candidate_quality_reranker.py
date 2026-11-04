"""Opt-in candidate-quality reranker experiment; production detector untouched.

Compete among strictly decodable top-three L2 candidates. Language coherence
must exceed the byte-margin evidence; reject candidates whose decoded Unicode
contains obvious control characters or non-text glyphs. Unlike codec-specific
exceptions, this test is family-agnostic and falsifiable on held-out folds.
"""
from __future__ import annotations
from dataclasses import dataclass
import math,unicodedata
from codeccat.vnext_l3_language import Rerank,LanguageCoherence,_decode

def text_defect_rate(text):
    if not text:return 1.0
    # U+FFFD, C0/C1 controls (excluding common whitespace), surrogates,
    # private-use or unassigned glyphs and box/block-drawing are suspicious.
    bad=0
    for ch in text:
        u=ord(ch)
        cat=unicodedata.category(ch)
        if ch in "\t\n\r":continue
        if ch=="\ufffd" or cat in ("Cc","Cs","Co","Cn") or 0x2500<=u<=0x259f:
            bad+=1
    return bad/len(text)

@dataclass(frozen=True)
class QualityReranker:
    language:LanguageCoherence
    max_margin:float=.35
    min_advantage:float=.25
    def rerank(self, ranked, data, *, final=True, max_l2_margin=.5,min_language_advantage=.15):
        base=ranked.encoding
        if base is None or len(ranked.candidates)<2:return Rerank(base,False,False,"no-contender")
        original=_decode(data,base,final=final)
        if original is None:return Rerank(base,True,False,"baseline-incomplete")
        base_language=self.language.score(original)
        base_defects=text_defect_rate(original)
        if base_language is None:return Rerank(base,False,False,"short-text")
        candidates=[]
        for i,codec in enumerate(ranked.candidates[1:3],start=1):
            margin=ranked.scores[0]-ranked.scores[i]
            if not (math.isfinite(margin) and 0<=margin<=self.max_margin):continue
            alt=_decode(data,codec,final=final)
            if alt is None or alt==original:continue
            score=self.language.score(alt)
            if score is None:continue
            advantage=score-base_language
            defects=text_defect_rate(alt)
            if advantage<=self.min_advantage:continue
            # Evidence gate. A seemingly fluent answer with extra controls
            # is not acceptable; avoid making invalid data look plausible.
            if defects>base_defects+0.005:continue
            candidates.append((advantage,codec))
        if not candidates:return Rerank(base,True,False,"no-quality-supported-alternative")
        advantage,codec=max(candidates)
        return Rerank(codec,True,True,"quality-coherence",advantage)
