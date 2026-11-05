"""Offline CodecCat trainer: safe dataset reading, group isolation, CCM1 export.

No dependency on codeccat's public runtime or internal legacy R21 training.
Users can install codeccat-train only in their training environment.
"""
from __future__ import annotations
import argparse,codecs,hashlib,json,struct
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

def read_manifest(manifest:Path):
    root=manifest.resolve().parent
    result=[];digests={};groups={}
    for number,line in enumerate(manifest.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():continue
        try:record=json.loads(line)
        except json.JSONDecodeError as ex:raise ValueError(f"line {number}: invalid JSON") from ex
        for key in ("path","encoding","group","split"):
            if not isinstance(record.get(key),str) or not record[key]:
                raise ValueError(f"line {number}: missing {key}")
        if record["split"] not in ("train","validation"):
            raise ValueError(f"line {number}: split must be train or validation")
        rel=Path(record["path"])
        resolved=(root/rel).resolve()
        if rel.is_absolute() or not resolved.is_relative_to(root):
            raise ValueError(f"line {number}: path escapes manifest directory")
        raw=resolved.read_bytes()
        if not raw:raise ValueError(f"line {number}: empty input")
        try:codecs.lookup(record["encoding"])
        except LookupError as ex:raise ValueError(f"line {number}: unknown codec") from ex
        try:raw.decode(record["encoding"],"strict")
        except UnicodeError as ex:raise ValueError(f"line {number}: label does not decode bytes") from ex
        sha=hashlib.sha256(raw).hexdigest()
        if sha in digests:
            old=digests[sha]
            if old!=(record["encoding"],record["split"]):
                raise ValueError(f"line {number}: duplicate bytes cross labels/splits")
            continue
        grp=record["group"];split=record["split"]
        if grp in groups and groups[grp]!=split:
            raise ValueError(f"line {number}: source-group leakage")
        groups[grp]=split
        digests[sha]=(record["encoding"],split)
        result.append((raw,record["encoding"],grp,split,sha))
    train=[r for r in result if r[3]=="train"]
    val=[r for r in result if r[3]=="validation"]
    if not train or not val:raise ValueError("both train and validation records required")
    if len({r[1] for r in train})<2:raise ValueError("need >=2 training codec labels")
    if any(r[1] not in {t[1] for t in train} for r in val):
        raise ValueError("validation codec label absent from training")
    return train,val

def write_string(parts,s):
    payload=s.encode("utf-8")
    if not 0<len(payload)<=256:raise ValueError("invalid model string length")
    parts.extend((struct.pack("<H",len(payload)),payload))

def export_ccm(path,vectorizer,classifier):
    vocab=vectorizer.vocabulary_
    width=len(vectorizer.idf_)
    classes=[str(x) for x in classifier.classes_]
    rows=1 if len(classes)==2 else len(classes)
    if len(vocab)!=width or classifier.coef_.shape!=(rows,width):
        raise ValueError("model tensor dimensions invalid")
    if classifier.intercept_.shape!=(rows,):
        raise ValueError("model bias dimensions invalid")
    if width>200_000 or len(classes)>256:raise ValueError("model exceeds native limit")
    low,high=vectorizer.ngram_range
    flags=int(vectorizer.binary)|2*int(vectorizer.sublinear_tf)
    parts=[struct.pack("<4sIIBBBB",b"CCM1",width,len(classes),low,high,flags,0)]
    for c in classes:write_string(parts,c)
    for gram,index in sorted(vocab.items(),key=lambda item:item[1]):
        parts.append(struct.pack("<I",index))
        write_string(parts,gram)
    for tensor in (vectorizer.idf_,classifier.intercept_,classifier.coef_):
        parts.append(np.asarray(tensor,dtype="<f8").tobytes(order="C"))
    data=b"".join(parts)
    data+=hashlib.sha256(data).digest()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(),len(data)

def fit(manifest:Path,output:Path,report:Path,seed:int):
    train,validation=read_manifest(manifest)
    vec=TfidfVectorizer(analyzer="char",ngram_range=(2,3),
        min_df=2,max_features=45000,sublinear_tf=True)
    strings=[r[0][:4096].decode("latin1") for r in train]
    x=vec.fit_transform(strings)
    model=SGDClassifier(loss="modified_huber",alpha=.0001,
        random_state=seed,max_iter=1500)
    model.fit(x,[r[1] for r in train])
    xval=vec.transform([r[0][:4096].decode("latin1") for r in validation])
    pred=model.predict(xval)
    accuracy=accuracy_score([r[1] for r in validation],pred)
    digest,size=export_ccm(output,vec,model)
    result={
        "schema":"codeccat-offline-train-v1",
        "feature_profile":"stable23-equivalent",
        "model_schema":"CCM1",
        "seed":seed,
        "training_n":len(train),"validation_n":len(validation),
        "training_groups":len({r[2] for r in train}),
        "validation_groups":len({r[2] for r in validation}),
        "classes":list(model.classes_),
        "validation_codec_label_accuracy":float(accuracy),
        "note":"Validation checks exact encoding LABEL, not full public Runtime text-level detection and not independent external generalization.",
        "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model_sha256":digest,
        "model_bytes":size
    }
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser(description="Offline-only CodecCat training; exports numeric CCM1 for Native Runtime")
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261015)
    a=p.parse_args()
    print("CODECCAT_TRAIN_RESULT",json.dumps(fit(a.manifest,a.output,a.report,a.seed)),flush=True)

if __name__=="__main__":main()
