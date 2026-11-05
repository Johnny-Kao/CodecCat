"""Zero-NumPy native L2 scorer with the established L1 ranking semantics.

Copied from the frozen public Alpha ranking implementation with only its
linear scoring backend replaced by an equivalent Rust calculation.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import codecs
from ._codeccat_native import NativeL2
from .vnext_gate import evaluate
from .vnext_bom import inspect_bom
from .vnext_escape import select_escape_encoding
from .vnext_unicode_structure import select_utf32_no_bom

@dataclass(frozen=True)
class Ranked:
    encoding: str | None
    candidates: tuple[str, ...]
    scores: tuple[float, ...]
    bom_hint: str | None
    ambiguous: bool

class TrainableDetector:
    """Inference-only model. Training happens in a separate environment."""
    def __init__(self,native:NativeL2,fingerprint:str,*,classes=None):
        self._native=native
        self.classifier=SimpleNamespace(classes_=tuple(classes if classes is not None else native.class_names()))
        self.fingerprint=fingerprint

    @classmethod
    def load_native(cls,path:str|Path):
        native=NativeL2.from_model_file(str(path))
        import hashlib
        digest=hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda:stream.read(65536),b""):
                digest.update(chunk)
        fp=digest.hexdigest()
        metadata=Path(str(path)+".meta.json")
        if metadata.is_file():
            import json
            info=json.loads(metadata.read_text(encoding="utf-8"))
            training_fp=info.get("training_fingerprint")
            if (info.get("schema")!="codeccat-native-model-provenance-v1"
                    or info.get("model_sha256")!=fp
                    or not isinstance(training_fp,str) or len(training_fp)!=64
                    or any(c not in "0123456789abcdef" for c in training_fp)):
                raise ValueError("invalid or mismatched native model provenance")
            fp=training_fp
        return cls(native,fp)

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
            # A BOM is decisive endian evidence. Never substitute the
            # opposite-endian codec merely because it shares the UTF family:
            # strict decoding can succeed while silently corrupting text.
            # A UTF-8 BOM is compatible with both utf-8-sig and utf-8;
            # decoding with utf-8 preserves the BOM as U+FEFF for the
            # public bom_policy layer to handle. UTF-16/32 must NOT fall
            # back to the opposite byte order.
            utf8_fallback = [
                str(name) for name in self.classifier.classes_
                if signature.encoding == "utf-8-sig"
                and str(name).replace("_", "-") == "utf-8"
            ]
            supported = generic if generic else (exact if exact else utf8_fallback)
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
                            # A legal UTF-16 surrogate pair is positive
                            # structural evidence for the opposite endian.
                            # Do not force a conflicting endian on NUL parity
                            # alone when it decodes supplementary characters.
                            opposite=("utf-16-le" if suggested=="utf-16-be"
                                      else "utf-16-be")
                            try:
                                alt=codecs.getincrementaldecoder(opposite)("strict")
                                alternative_text=alt.decode(prefix,final=final and len(data)<=4096)
                                opposite_astral=any(ord(ch)>0xffff for ch in alternative_text)
                            except (UnicodeError,LookupError):
                                opposite_astral=False
                            if not opposite_astral:
                                # BOM-less UTF-32 astral values can exhibit
                                # the same alternating-NUL parity as UTF-16.
                                # A valid UTF-32 interpretation makes this
                                # structurally ambiguous, not a forced match.
                                other=("utf-32-be" if even_zeros>odd_zeros
                                       else "utf-32-le")
                                try:
                                    dec32=codecs.getincrementaldecoder(other)("strict")
                                    dec32.decode(prefix,final=final and len(data)<=4096)
                                except (UnicodeError,LookupError):
                                    return Ranked(suggested,(suggested,),(1.0,),None,False)
                            # Structural evidence is ambiguous: defer to L2.
        # Opt-in integrated L1 -> L2 path. A codec is filtered ONLY on
        # strict decoding impossibility, never on family plausibility.
        # Baseline remains unchanged until latency and paired accuracy gates.
        eligible = None
        if structural_first:
            evidence = self.analyze_structure(data, final=final)
            eligible = set(evidence.possible)
            if not eligible:
                return Ranked(None, tuple(), tuple(), signature.encoding, True)
        score=self._native.scores(data)
        classes=list(self.classifier.classes_)
        # Stable class-order tie handling is required for user-facing
        # candidate order. Ordinarily no exact L2 ties occur.
        order=sorted(range(len(score)), key=lambda i:(score[i],i), reverse=True)
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


