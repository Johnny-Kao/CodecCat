"""Offline export of an already trained CodecCat L2 model to native CCM1.

Requires NumPy/scikit-learn ONLY during training/export. The resulting
read-only artifact is loaded directly by Rust without those libraries.
"""
from __future__ import annotations
import argparse,hashlib,struct
from pathlib import Path
from codeccat.vnext_strong_l2 import TrainableDetector

def write_string(parts,value):
    data=value.encode("utf-8")
    if not 0<len(data)<=256:raise ValueError("invalid model string length")
    parts.append(struct.pack("<H",len(data)))
    parts.append(data)

def export(source:Path,destination:Path,precision:str="f64"):
    model=TrainableDetector.load_portable(source)
    vec=model.vectorizer
    clf=model.classifier
    width=len(vec.idf_)
    classes=[str(x) for x in clf.classes_]
    if len(vec.vocabulary_)!=width:raise ValueError("bad source vocabulary")
    if clf.coef_.shape!=(len(classes),width):
        raise ValueError("binary/source tensor shape not supported by CCM1")
    min_gram,max_gram=vec.ngram_range
    flags=int(vec.binary)|(2 if vec.sublinear_tf else 0)
    if precision not in ("f64","f32"):raise ValueError("unsupported precision")
    schema=b"CCM2" if precision=="f32" else b"CCM1"
    parts=[struct.pack("<4sIIBBBB",schema,width,len(classes),min_gram,max_gram,flags,0)]
    for c in classes:write_string(parts,c)
    for gram,index in sorted(vec.vocabulary_.items(),key=lambda item:item[1]):
        parts.append(struct.pack("<I",index))
        write_string(parts,str(gram))
    for array in (vec.idf_,clf.intercept_):
        parts.append(array.astype("<f8",copy=False).tobytes(order="C"))
    parts.append(clf.coef_.astype("<f4" if precision=="f32" else "<f8",copy=False).tobytes(order="C"))
    payload=b"".join(parts)
    payload+=hashlib.sha256(payload).digest()
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(payload)
    print("NATIVE_CCM1_EXPORT",str(destination),len(payload),
          hashlib.sha256(payload).hexdigest(),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=Path("src/codeccat/models/stable23.zip"))
    p.add_argument("--output",type=Path,default=Path("native-prototype/model/stable23.ccm"))
    p.add_argument("--precision",choices=("f64","f32"),default="f64")
    args=p.parse_args()
    export(args.source,args.output,args.precision)
