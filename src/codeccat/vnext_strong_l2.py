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
            supported=[str(name) for name in self.classifier.classes_
                       if str(name).replace("_","-").startswith(family)]
            if not supported:
                return Ranked(None,tuple(),tuple(),signature.encoding,True)
            if len(supported)==1:
                label=supported[0]
                try:
                    data.decode(label,"strict")
                except (UnicodeError,LookupError):
                    return Ranked(None,tuple(),tuple(),signature.encoding,True)
                return Ranked(label,(label,),(1.0,),signature.encoding,False)
        x=self.vectorizer.transform([data[:4096].decode("latin1")])
        raw=self.classifier.decision_function(x)
        # scikit-learn binary case emits one signed margin.
        margins=np.asarray(raw)
        if margins.ndim==1:
            score=np.array([-float(margins[0]), float(margins[0])])
        else:score=margins[0]
        classes=list(self.classifier.classes_)
        order=np.argsort(score)[::-1]
        gate=evaluate(data[:4096],tuple(classes),final=final and len(data)<=4096)
        compatible=set(gate.possible)
        bom=gate.bom.encoding
        if bom:
            base="-".join(bom.split("-")[:2])
            allowed={c for c in compatible if c.replace("_","-").startswith(base)}
            if not allowed:
                return Ranked(None,tuple(),tuple(),bom,True)
            compatible=allowed
        filtered=[int(i) for i in order if classes[int(i)] in compatible]
        if not filtered:return Ranked(None,tuple(),tuple(),bom,True)
        top=filtered[0]
        ordered=tuple(classes[i] for i in filtered)
        scores=tuple(float(score[i]) for i in filtered)
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
