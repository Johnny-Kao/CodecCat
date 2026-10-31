"""Experimental vNext Feature Schema: fixed logic evidence and ordered bytes.

No mutable parameters or model weights here; both training and future inference
must execute this same extractor before any parameterized decision policy.
"""
import numpy as np
from .features import baseline_from_array,hmt768_array

SCHEMA="vnext-evidence-ordered-1"
ORDERED_BINS=512
N_EVIDENCE=10
CODECS=("utf-8","shift_jis","euc_jp","big5","gb18030")

def _decode(head,codec):
    for trim in range(4):
        if len(head)<=trim:break
        try:return head[:len(head)-trim].decode(codec,"strict")
        except UnicodeDecodeError:pass
    return None

def evidence(data):
    head=data[:4096]
    dec={codec:_decode(head,codec) for codec in CODECS}
    flags=[1.0 if dec[c] is not None else 0.0 for c in CODECS]
    def fraction(s,lo,hi):
        if not s:return 0.0
        return sum(lo<=c<=hi for c in s)/len(s)
    sj=dec["shift_jis"];ej=dec["euc_jp"]
    kana=lambda s:fraction(s,"\u3040","\u309f")+fraction(s,"\u30a0","\u30ff")
    sjk=kana(sj);ejk=kana(ej)
    # Decode-on-demand semantics for single-byte scripts; no label oracle.
    cyr=fraction(head.decode("cp1251","replace"),"\u0400","\u04ff") if head else 0.0
    ara=fraction(head.decode("cp1256","replace"),"\u0600","\u06ff") if head else 0.0
    return np.asarray(flags+[sjk,ejk,sjk-ejk,cyr,ara],dtype=np.float32)

def ordered(data):
    a=hmt768_array(data)
    vec=np.zeros(ORDERED_BINS,dtype=np.float32)
    if len(a)>=2:
        x=a[:-1].astype(np.uint16)
        y=a[1:].astype(np.uint16)
        ids=((x*131)^(y*17))&511
        vec[:]=np.bincount(ids.astype(np.intp),minlength=ORDERED_BINS)/(len(a)-1)
    return vec

def feature_variants(data):
    base=baseline_from_array(hmt768_array(data))
    logic=evidence(data)
    pair=ordered(data)
    assert len(logic)==N_EVIDENCE and len(pair)==ORDERED_BINS
    return {"base":base,
            "base_logic":np.concatenate((base,logic)),
            "base_ordered":np.concatenate((base,pair)),
            "combined":np.concatenate((base,logic,pair))}
