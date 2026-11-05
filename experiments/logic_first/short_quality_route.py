"""Short-input decoded-text anomaly guard (research-only).

The length check is at the very start of rank(). For <=256 bytes only,
consider lower-ranked *trained* encodings when the baseline decode contains
multiple C1 controls, private-use, or unassigned Unicode codepoints.
Never infer a codec from filenames, source, ground-truth labels or competitor
outputs. No changes to released runtime or default classifier.
"""
from __future__ import annotations
import codecs
import unicodedata
import numpy as np
from codeccat.vnext_strong_l2 import Ranked

SHORT_LIMIT=256
SEARCH_DEPTH=16
MIN_SUSPICIOUS=2
MIN_SUSPICIOUS_FRACTION=0.02

def decode(raw,codec):
    try:return codecs.getincrementaldecoder(codec)("strict").decode(raw,final=True)
    except (LookupError,UnicodeError,ValueError):return None

def suspicious(text):
    if not text:return 0
    return sum(
        ((0x80<=ord(c)<=0x9f) or (unicodedata.category(c) in ("Co","Cn","Cs")))
        for c in text
    )

class ShortQualityRanker:
    def __init__(self,model):
        self.model=model
    def rank(self,data,*,final=True,structural_first=False):
        base=self.model.rank(data,final=final,structural_first=structural_first)
        # Never touch long text, hard-coded byte signatures or singletons.
        if len(data)>SHORT_LIMIT or len(base.candidates)<2 or base.encoding is None:
            return base
        original=decode(data,str(base.encoding))
        if original is None:return base
        anomaly=suspicious(original)
        if anomaly<MIN_SUSPICIOUS or anomaly/max(1,len(original))<MIN_SUSPICIOUS_FRACTION:
            return base
        # Use only model classes. Enumerate by trained L2 margin; do not
        # run a broader codec detector or boost particular encoding families.
        x=self.model.vectorizer.transform([data.decode("latin1")])
        clf=self.model.classifier
        values=np.asarray(clf.decision_function(x)).reshape(-1)
        classes=tuple(str(v) for v in clf.classes_)
        if len(classes)==2:
            values=np.array([-values[0],values[0]])
        order=np.argsort(values)[::-1]
        viable=[]
        for j in order:
            codec=classes[int(j)]
            text=decode(data,codec)
            if text is None:continue
            viable.append((codec,float(values[int(j)]),suspicious(text)))
            if len(viable)>=SEARCH_DEPTH:break
        for codec,score,issues in viable:
            if issues==0 and codec!=base.encoding:
                return Ranked(codec,(codec,)+tuple(c for c in base.candidates if c!=codec),
                              (score,)+tuple(s for c,s in zip(base.candidates,base.scores) if c!=codec),
                              base.bom_hint,base.ambiguous)
        return base
