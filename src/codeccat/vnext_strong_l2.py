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
from .vnext_escape import select_escape_encoding
from .vnext_unicode_structure import select_utf32_no_bom

SCHEMA = "vnext-l2-tfidf-ordered23-sgd-huber-v1"
SCHEMA_V2 = "vnext-l2-tfidf-ordered13-sgd-huber-v2"

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
    def fit(cls, samples: Iterable[tuple[bytes, str, str]], *, feature_profile: str = "stable23") -> "TrainableDetector":
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
        if feature_profile not in ("stable23", "experimental13"):
            raise ValueError("unknown feature profile")
        params = ((2,3),2,45000) if feature_profile=="stable23" else ((1,3),1,90000)
        vec=TfidfVectorizer(analyzer="char",ngram_range=params[0],min_df=params[1],
                            max_features=params[2],sublinear_tf=True)
        features=vec.fit_transform(strings)
        clf=SGDClassifier(loss="modified_huber",alpha=0.0001,
                          random_state=20261015,max_iter=1500)
        clf.fit(features,labels)
        return cls(vectorizer=vec,classifier=clf,fingerprint=fingerprint.hexdigest())

    def analyze_structure(self, data: bytes, *, final: bool = True):
        """L1-only evidence for all supported classes; no model scoring.

        Analyze the actual inference prefix. A truncated 4096-byte prefix
        cannot prove that a codec is invalid merely due to an incomplete
        final character. The gate retains aliases as separate trained labels.
        """
        if not isinstance(data, bytes):
            raise TypeError("expected bytes")
        prefix = data[:4096]
        return evaluate(
            prefix, tuple(str(name) for name in self.classifier.classes_),
            final=final and len(data) <= 4096,
        )

    def rank(self,data:bytes,*,final:bool=True,structural_first:bool=False) -> Ranked:
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
                    decoder=codecs.getincrementaldecoder(label)("strict")
                    decoder.decode(data[:4096],final=final and len(data)<=4096)
                except (UnicodeError,LookupError):
                    return Ranked(None,tuple(),tuple(),signature.encoding,True)
                return Ranked(label,(label,),(1.0,),signature.encoding,False)
        escape_label=select_escape_encoding(
            data,tuple(str(x) for x in self.classifier.classes_),final=final)
        if escape_label is not None:
            return Ranked(escape_label,(escape_label,),(1.0,),None,False)
        utf32=select_utf32_no_bom(
            data,tuple(str(x) for x in self.classifier.classes_),final=final)
        if utf32 is not None:
            return Ranked(utf32,(utf32,),(1.0,),None,False)
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
        # Opt-in integrated L1 -> L2 path. A codec is filtered ONLY on
        # strict decoding impossibility, never on family plausibility.
        # Baseline remains unchanged until latency and paired accuracy gates.
        eligible = None
        if structural_first:
            evidence = self.analyze_structure(data, final=final)
            eligible = set(evidence.possible)
            if not eligible:
                return Ranked(None, tuple(), tuple(), signature.encoding, True)
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
            if eligible is not None:
                if label not in eligible:
                    continue
            else:
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

    def save_portable(self, path: str | Path) -> None:
        """Non-executable model archive: JSON schema and numeric NumPy arrays.

        Do not treat any untrusted model as authoritative even when parsing is safe.
        """
        import io
        import zipfile
        out = Path(path)
        vocab = self.vectorizer.vocabulary_
        meta = {
            "schema": SCHEMA if self.vectorizer.ngram_range == (2,3) else SCHEMA_V2,
            "fingerprint": self.fingerprint,
            "vocabulary": {str(k): int(v) for k, v in vocab.items()},
            "classes": [str(x) for x in self.classifier.classes_],
            "params": {
                "ngram_range": list(self.vectorizer.ngram_range),
                "sublinear_tf": bool(self.vectorizer.sublinear_tf),
                "lowercase": bool(self.vectorizer.lowercase),
                "norm": self.vectorizer.norm,
            },
        }
        arrays = io.BytesIO()
        np.savez_compressed(
            arrays,
            idf=np.asarray(self.vectorizer.idf_, dtype=np.float64),
            coef=np.asarray(self.classifier.coef_, dtype=np.float64),
            intercept=np.asarray(self.classifier.intercept_, dtype=np.float64),
        )
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(meta, sort_keys=True, separators=(",", ":")))
            archive.writestr("weights.npz", arrays.getvalue())

    @classmethod
    def load_portable(cls, path: str | Path) -> "TrainableDetector":
        """Load numeric-only archive; never unpickle a model bundle."""
        import io
        import zipfile
        from sklearn.feature_extraction.text import TfidfTransformer
        with zipfile.ZipFile(path, "r") as archive:
            if set(archive.namelist()) != {"manifest.json", "weights.npz"}:
                raise ValueError("unexpected model archive entries")
            if archive.getinfo("manifest.json").file_size > 8_000_000:
                raise ValueError("manifest is too large")
            if archive.getinfo("weights.npz").file_size > 200_000_000:
                raise ValueError("weights are too large")
            meta = json.loads(archive.read("manifest.json"))
            schema = meta.get("schema")
            if schema not in (SCHEMA, SCHEMA_V2):
                raise ValueError("model schema mismatch")
            params = meta["params"]
            expected_range = [2, 3] if schema == SCHEMA else [1, 3]
            if params != {
                "ngram_range": expected_range,
                "sublinear_tf": True,
                "lowercase": True,
                "norm": "l2",
            }:
                raise ValueError("unsupported vectorizer configuration")
            with np.load(io.BytesIO(archive.read("weights.npz")), allow_pickle=False) as tensors:
                if set(tensors.files) != {"idf", "coef", "intercept"}:
                    raise ValueError("unexpected model tensors")
                idf = tensors["idf"].copy()
                coef = tensors["coef"].copy()
                intercept = tensors["intercept"].copy()
        vocab = meta["vocabulary"]
        classes = np.asarray(meta["classes"], dtype=str)
        if (not isinstance(vocab, dict) or not vocab or
                sorted(vocab.values()) != list(range(len(vocab))) or
                idf.shape != (len(vocab),) or coef.ndim != 2 or
                coef.shape[1] != len(vocab) or
                intercept.shape != (coef.shape[0],) or
                coef.shape[0] != (1 if len(classes) == 2 else len(classes)) or
                len(classes) < 2 or
                not np.isfinite(idf).all() or not np.isfinite(coef).all() or
                not np.isfinite(intercept).all()):
            raise ValueError("invalid portable model shape or values")
        vec = TfidfVectorizer(
            analyzer="char", ngram_range=tuple(expected_range),
            vocabulary=vocab, sublinear_tf=True,
        )
        vec.fixed_vocabulary_ = True
        vec.vocabulary_ = dict(vocab)
        vec._tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
        vec._tfidf.idf_ = idf
        clf = SGDClassifier(loss="modified_huber")
        clf.classes_ = classes
        clf.coef_ = coef
        clf.intercept_ = intercept
        return cls(vectorizer=vec, classifier=clf, fingerprint=str(meta["fingerprint"]))
