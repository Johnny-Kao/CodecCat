"""Offline CodecCat trainer: safe dataset reading, group isolation, CCM1 export.

No dependency on codeccat's public runtime or internal legacy R21 training.
Users can install codeccat-train only in their training environment.
"""
from __future__ import annotations
import argparse,codecs,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

def read_manifest(manifest:Path,require_validation:bool=True):
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
        # Optional per-file attestations protect reproducible training against
        # silent input changes after a manifest was generated.
        for key,payload in (("sha256",raw),("prefix_sha256",raw[:4096])):
            expected=record.get(key)
            if expected is not None:
                actual=hashlib.sha256(payload).hexdigest()
                if not isinstance(expected,str) or expected!=actual:
                    raise ValueError(f"line {number}: {key} mismatch")
        try:codecs.lookup(record["encoding"])
        except LookupError as ex:raise ValueError(f"line {number}: unknown codec") from ex
        try:raw.decode(record["encoding"],"strict")
        except UnicodeError as ex:raise ValueError(f"line {number}: label does not decode bytes") from ex
        sha=hashlib.sha256(raw).hexdigest()
        grp=record["group"];split=record["split"]
        # Register and validate groups before deduplicating identical content.
        # A duplicate can introduce a group that later appears in another split.
        if grp in groups and groups[grp]!=split:
            raise ValueError(f"line {number}: source-group leakage")
        groups[grp]=split
        if sha in digests:
            old=digests[sha]
            if old!=(record["encoding"],split):
                raise ValueError(f"line {number}: duplicate bytes cross labels/splits")
            continue
        digests[sha]=(record["encoding"],split)
        result.append((raw,record["encoding"],grp,split,sha))
    train=[r for r in result if r[3]=="train"]
    val=[r for r in result if r[3]=="validation"]
    if not train or (require_validation and not val):
        raise ValueError("both train and validation records required")
    if len({r[1] for r in train})<2:raise ValueError("need >=2 training codec labels")
    if any(r[1] not in {t[1] for t in train} for r in val):
        raise ValueError("validation codec label absent from training")
    return train,val

def fingerprint_dataset(rows):
    """Fingerprint accepted labeled bytes and assignments without storing texts."""
    source_rows = [
        {"sha256": row[4], "encoding": row[1], "group": row[2], "split": row[3]}
        for row in rows
    ]
    return hashlib.sha256(json.dumps(
        source_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()

def historical_stable23_fingerprint(rows):
    """Match the original TrainableDetector fingerprint on 4KB corpus prefixes.

    This fingerprint is an audit value, not a trust or license certificate.
    """
    digest=hashlib.sha256()
    for row in rows:
        raw,encoding,group=row[:3]
        prefix_hash=hashlib.sha256(raw[:4096]).hexdigest()
        digest.update(prefix_hash.encode()+b"\0"+encoding.encode()+b"\0"+group.encode()+b"\0")
    return digest.hexdigest()

def write_string(parts,s):
    payload=s.encode("utf-8")
    if not 0<len(payload)<=256:raise ValueError("invalid model string length")
    parts.extend((struct.pack("<H",len(payload)),payload))

def export_ccm(path,vectorizer,classifier,precision="f64"):
    if precision not in ("f64", "f32"):
        raise ValueError("unsupported weight precision")
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
    schema=b"CCM2" if precision=="f32" else b"CCM1"
    parts=[struct.pack("<4sIIBBBB",schema,width,len(classes),low,high,flags,0)]
    for c in classes:write_string(parts,c)
    for gram,index in sorted(vocab.items(),key=lambda item:item[1]):
        parts.append(struct.pack("<I",index))
        write_string(parts,gram)
    for tensor in (vectorizer.idf_, classifier.intercept_):
        parts.append(np.asarray(tensor,dtype="<f8").tobytes(order="C"))
    weights_dtype="<f4" if precision=="f32" else "<f8"
    parts.append(np.asarray(classifier.coef_,dtype=weights_dtype).tobytes(order="C"))
    data=b"".join(parts)
    data+=hashlib.sha256(data).digest()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(),len(data)

def fit_l2(training_rows,seed:int):
    """Stable23-equivalent L2 fitting on supplied rows; no official corpus implied."""
    strings=[r[0][:4096].decode("latin1") for r in training_rows]
    labels=[r[1] for r in training_rows]
    vec=TfidfVectorizer(analyzer="char",ngram_range=(2,3),
        min_df=2,max_features=45000,sublinear_tf=True)
    x=vec.fit_transform(strings)
    model=SGDClassifier(loss="modified_huber",alpha=.0001,
        random_state=seed,max_iter=1500)
    model.fit(x,labels)
    return vec,model

def fit(manifest:Path,output:Path,report:Path,seed:int,precision:str="f64",residual_focus:bool=False,train_only:bool=False):
    train,validation=read_manifest(manifest,require_validation=not train_only)
    if train_only and validation:
        raise ValueError("--train-only requires a manifest containing train rows only")
    original_train_n=len(train)
    source_fingerprint=fingerprint_dataset(train + validation)
    removed={}
    if residual_focus:
        from .residual import residual_subset
        train,removed=residual_subset(train)
    effective_fingerprint=fingerprint_dataset(train)
    vec,model=fit_l2(train,seed)
    accuracy=None
    if validation:
        xval=vec.transform([r[0][:4096].decode("latin1") for r in validation])
        pred=model.predict(xval)
        accuracy=float(accuracy_score([r[1] for r in validation],pred))
    digest,size=export_ccm(output,vec,model,precision=precision)
    result={
        "schema":"codeccat-offline-train-v1",
        "feature_profile":"stable23-equivalent",
        "model_schema":"CCM2" if precision=="f32" else "CCM1",
        "weight_precision":precision,
        "seed":seed,
        "training_n":len(train),"validation_n":len(validation),
        "training_original_n":original_train_n,
        "residual_focus_applied":residual_focus,
        "training_removed_by_label":removed,
        "training_groups":len({r[2] for r in train}),
        "validation_groups":len({r[2] for r in validation}),
        "classes":list(model.classes_),
        "validation_codec_label_accuracy":accuracy,
        "train_only":train_only,
        "note":"Validation checks exact encoding LABEL, not full public Runtime text-level detection and not independent external generalization.",
        "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "dataset_content_fingerprint_sha256":source_fingerprint,
        "effective_training_fingerprint_sha256":effective_fingerprint,
        "historical_stable23_fingerprint":historical_stable23_fingerprint(train),
        "dataset_fingerprint_scope":"SHA256 of ordered accepted bytes SHA256, encoding, group and split; no raw text",
        "model_sha256":digest,
        "model_bytes":size,
        "training_tool_version":"0.1.0a3+source",
        "python_version":sys.version.split()[0],
        "numpy_version":np.__version__,
        "sklearn_version":__import__("sklearn").__version__,
        "training_config":{"min_df":2,"max_features":45000,"ngram_range":[2,3],
                           "sublinear_tf":True,"loss":"modified_huber",
                           "alpha":0.0001,"max_iter":1500,
                           "byte_limit_per_document":4096}
    }
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return result

def main():
    import sys
    args=sys.argv[1:]
    if args and args[0]=="evaluate":
        from .evaluate import evaluate
        p=argparse.ArgumentParser(description="Compare a custom model against the bundled Stable model on labeled bytes")
        p.add_argument("--manifest",type=Path,required=True)
        p.add_argument("--model",type=Path,required=True)
        p.add_argument("--report",type=Path,required=True)
        p.add_argument("--baseline-model",type=Path,default=None)
        p.add_argument("--split",choices=("test","validation"),default="test")
        a=p.parse_args(args[1:])
        print("CODECCAT_MODEL_COMPARISON",
              json.dumps(evaluate(a.manifest,a.model,a.report,a.split,a.baseline_model),sort_keys=True),
              flush=True)
        return
    # Preserve the original CLI without subcommands for existing users.
    if args and args[0]=="fit":
        args=args[1:]
    p=argparse.ArgumentParser(description="Offline-only CodecCat training; exports numeric CCM1/CCM2 for Native Runtime")
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261015)
    p.add_argument("--train-only",action="store_true",
                   help="fit every labeled row; requires train-only manifest and reports no validation accuracy")
    p.add_argument("--residual-focus",action="store_true",
                   help="replay historical Stable23 residual sample filtering (not official release approval)")
    p.add_argument("--precision",choices=("f64","f32"),default="f64",
                   help="f64 exports CCM1 (default); f32 exports compact CCM2 weights")
    a=p.parse_args(args)
    print("CODECCAT_TRAIN_RESULT",json.dumps(fit(a.manifest,a.output,a.report,a.seed,precision=a.precision,residual_focus=a.residual_focus,train_only=a.train_only)),flush=True)

if __name__=="__main__":main()
