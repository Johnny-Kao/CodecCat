"""Independent trainable vNext L2 adapter.

Optional scikit-learn dependency. Does not import legacy Runtime / R21.
Only load locally trusted joblib bundles (joblib uses pickle).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import codecs

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from .vnext_gate import evaluate
from .vnext_bom import inspect_bom

SCHEMA = "vnext-l2-tfidf-ordered23-sgd-huber-v1"

@dataclass(frozen=True)
class Ranked:
    encoding: str | None
    candidates: tuple[str, ...]
    scores: tuple[float, ...]
    bom_hint: str | None
    ambiguous: bool

class TrainableDetector:
    def __init__(self, *, vectorizer, classifier, fingerprint: str):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.fingerprint = fingerprint

    @classmethod
    def fit(cls, samples: Iterable[tuple[bytes, str, str]]) -> "TrainableDetector":
        rows = list(samples)
        if not rows:raise ValueError("empty dataset")
        labels, strings = [], []
        seen = {}
        fingerprint = hashlib.sha256()
        for data, encoding, source_group in rows:
            if not isinstance(data, bytes) or not encoding or not source_group:
                raise ValueError("expected (bytes, label, source_group)")
            digest=hashlib.sha256(data).hexdigest()
            if digest in seen and seen[digest]!=encoding:
                raise ValueError("conflicting labels for exact bytes")
            seen[digest]=encoding
            strings.append(data[:4096].decode("latin1"))
            labels.append(encoding)
            fingerprint.update(digest.encode()+b"\0"+encoding.encode()+b"\0"+source_group.encode()+b"\0")
        if len(set(labels)) < 2: raise ValueError("need at least two classes")
        vec=TfidfVectorizer(analyzer="char",ngram_range=(2,3),min_df=2,
                            max_features=45000,sublinear_tf=True)
        features=vec.fit_transform(strings)
        clf=SGDClassifier(loss="modified_huber",alpha=0.0001,
                          random_state=20261015,max_iter=1500)
        clf.fit(features,labels)
        return cls(vectorizer=vec,classifier=clf,fingerprint=fingerprint.hexdigest())

    def rank(self,data:bytes,*,final:bool=True) -> Ranked:
        if not isinstance(data,bytes):raise TypeError("expected bytes")
        # BOM evidence is stable and cheap: avoid trained scoring when decisive.
        signature=inspect_bom(data, final=final)
        if signature.incomplete:
            return Ranked(None,tuple(),tuple(),None,True)
        if signature.encoding:
            family="-".join(signature.encoding.split("-")[:2])
            # BOM-bearing generic UTF-16/32 streams use the generic codec
            # label when that class is supported. Explicit endian labels
            # remain eligible for BOM-less streams via the learned ranking.
            generic=[str(name) for name in self.classifier.classes_
                     if family in ("utf-16","utf-32") and str(name).replace("_","-")==family]
            exact=[str(name) for name in self.classifier.classes_
                   if str(name).replace("_","-")==signature.encoding]
            supported=generic if generic else (exact if exact else [
                str(name) for name in self.classifier.classes_
                if str(name).replace("_","-").startswith(family)])
            if not supported:
                return Ranked(None,tuple(),tuple(),signature.encoding,True)
            if len(supported)==1:
                label=supported[0]
                try:
                    data.decode(label,"strict")
                except (UnicodeError,LookupError):
                    return Ranked(None,tuple(),tuple(),signature.encoding,True)
                return Ranked(label,(label,),(1.0,),signature.encoding,False)
        # UTF-16 ASCII-heavy text without BOM has a strong byte-position
        # invariant: nulls cluster on the high byte. Only act with substantial
        # evidence; mixed/non-ASCII text falls through to trained L2.
        if not signature.encoding and len(data)>=32:
            prefix=data[:4096]
            pairs=len(prefix)//2
            if pairs:
                even_zeros=prefix[0:2*pairs:2].count(0)
                odd_zeros=prefix[1:2*pairs:2].count(0)
                dominant=max(even_zeros,odd_zeros)
                minor=min(even_zeros,odd_zeros)
                if dominant>=max(12,int(pairs*0.30)) and dominant>=4*max(1,minor):
                    suggested="utf-16-be" if even_zeros>odd_zeros else "utf-16-le"
                    if suggested in self.classifier.classes_:
                        try:
                            decoder=codecs.getincrementaldecoder(suggested)("strict")
                            decoder.decode(prefix,final=final and len(data)<=4096)
                        except (UnicodeError,LookupError):
                            pass
                        else:
                            return Ranked(suggested,(suggested,),(1.0,),None,False)
        x=self.vectorizer.transform([data[:4096].decode("latin1")])
        # Native sparse linear projection avoids sklearn's prediction-layer
        # overhead, while using EXACTLY the learned coefficients/intercept.
        row=x.getrow(0)
        weights=self.classifier.coef_[:,row.indices]
        margins=np.asarray(weights @ row.data + self.classifier.intercept_)
        # scikit-learn binary case stores one signed margin for class 1.
        if len(self.classifier.classes_) == 2:
            score=np.array([-float(margins[0]),float(margins[0])])
        else:
            score=margins
        classes=list(self.classifier.classes_)
        order=np.argsort(score)[::-1]
        # Lazy L1: visit candidates in learned-score order; validate the
        # contenders that can affect the outcome, not every trained codec.
        # Returned candidates are the validated TOP THREE, not the full
        # set of theoretically valid encodings.
        validated=[]
        prefix=data[:4096]
        for idx in order:
            label=str(classes[int(idx)])
            try:
                dec=codecs.getincrementaldecoder(label)("strict")
                dec.decode(prefix,final=final and len(data)<=4096)
            except (UnicodeError,LookupError):
                continue
            validated.append(int(idx))
            if len(validated)>=3:
                break
        if not validated:return Ranked(None,tuple(),tuple(),signature.encoding,True)
        top=validated[0]
        ordered=tuple(str(classes[i]) for i in validated)
        scores=tuple(float(score[i]) for i in validated)
        bom=signature.encoding
        # No uncalibrated numeric probability claims.
        return Ranked(classes[top],ordered,scores,bom,False)

    def save(self,path: str|Path) -> None:
        import joblib
        joblib.dump({"schema":SCHEMA,"fingerprint":self.fingerprint,
                     "vectorizer":self.vectorizer,"classifier":self.classifier},path)

    @classmethod
    def load_trusted(cls,path:str|Path) -> "TrainableDetector":
        """Only call for bundles created by a trusted source; joblib unpickles."""
        import joblib
        payload=joblib.load(path)
        if payload.get("schema")!=SCHEMA:raise ValueError("schema mismatch")
        return cls(vectorizer=payload["vectorizer"],classifier=payload["classifier"],
                   fingerprint=payload["fingerprint"])
