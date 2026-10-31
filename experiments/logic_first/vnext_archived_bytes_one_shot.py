"""One-time confirmation on independently archived, natively labeled byte fixtures.

The source/manifest was frozen in GitHub Actions run 41237703300 before
this model scoring script was written. Historical uchardet suite is test-only:
do not train from it, redistribute its bytes, or claim natural provenance.
"""
from __future__ import annotations
import argparse,codecs,collections,hashlib,json,pathlib,platform,subprocess,sys
from charset_normalizer import from_bytes
import charset_normalizer
import chardet
from codeccat.vnext_decoding import decode
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_selective import SelectiveLatinDetector

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_content_safe_contrast import contrastive_augmentation
from vnext_training_strategy_audit import strict_text
from vnext_archived_bytes_freeze import label_for

EXPECTED_FINGERPRINT="e23005af8b0f5cb85e6b02586de5fa5ba520cc3d2cdb3e826ec8d6267faffd98"
EXPECTED_CORPUS_REV="4e685757780cb3c652fc6c9ec759f62888969ec9"
EXPECTED_TRAIN_REV="b0c0d206913ba118ffea2f7d306883e7a4a76db7"
BASE_NAMES=("baseline","latin_only","routed","chardet7","charset_normalizer")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check_rev(root,expected):
    actual=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if actual!=expected: raise RuntimeError(f"unexpected pinned revision {actual}")
    return actual


def cases_from_manifest(root, manifest, training):
    data=json.loads(pathlib.Path(manifest).read_text())
    if data["fingerprint"]!=EXPECTED_FINGERPRINT or data["origin_revision"]!=EXPECTED_CORPUS_REV:
        raise RuntimeError("previously frozen source metadata has changed")
    rows, excluded, seen=set(),collections.Counter(),set()
    training_texts={sha(strict_text(raw,label).encode("utf-8","surrogatepass"))
                    for raw,label,group in training}
    for item in data["files"]:
        if item["classification"]!="valid_tagged_bytes":
            excluded[item["classification"]]+=1
            continue
        p=(pathlib.Path(root)/item["path"]).resolve()
        if not p.is_relative_to(pathlib.Path(root).resolve()):
            raise RuntimeError("path traversal")
        raw=p.read_bytes()
        if sha(raw)!=item["sha256"] or len(raw)!=item["bytes"]:
            raise RuntimeError("frozen raw fixture bytes changed")
        codec=item["python_codec"]
        reference=raw.decode(codec,"strict")
        tid=sha(reference.encode("utf-8","surrogatepass"))
        if tid!=item["decoded_sha256"]:
            raise RuntimeError("decoded content digest mismatch")
        if tid in training_texts:
            excluded["duplicate_training_document"]+=1
            continue
        if sha(raw) in seen:
            excluded["duplicate_test_raw_bytes"]+=1
            continue
        seen.add(sha(raw))
        # Some sources legitimately share identical decoded text across
        # multiple codec labels: same decoded text is the primary goal.
        rows.add((item["path"],item["lang"],codec,raw,reference))
    if len(rows)<35:
        raise RuntimeError(f"too few independent labeled bytes: {len(rows)}")
    return sorted(rows,key=lambda x:x[0]), dict(excluded), data["fingerprint"]


def third_party(name,raw):
    if name=="chardet7":
        label=chardet.detect(raw).get("encoding")
    else:
        result=from_bytes(raw).best()
        label=result.encoding if result is not None else None
    try:
        text=raw.decode(label,"strict") if label else None
    except (UnicodeError,LookupError,TypeError,ValueError):
        text=None
    return label,text


def scores(rows):
    metrics={}
    for name in BASE_NAMES:
        hits=sum(r["predictions"][name]["same_text"] for r in rows)
        labels=sum(r["predictions"][name]["exact_codec"] for r in rows)
        metrics[name]={"correct_decoded_text":hits,"exact_codec_label":labels,"n":len(rows)}
    paired={}
    for name in BASE_NAMES[1:]:
        helped=sum(r["predictions"][name]["same_text"] and
                   not r["predictions"]["baseline"]["same_text"] for r in rows)
        harmed=sum(r["predictions"]["baseline"]["same_text"] and
                   not r["predictions"][name]["same_text"] for r in rows)
        paired[name]={"helps":helped,"harms":harmed,"net":helped-harmed}
    groups={}
    for key in sorted({r["lang"] for r in rows}):
        subset=[r for r in rows if r["lang"]==key]
        groups[key]={"n":len(subset),**{name:sum(r["predictions"][name]["same_text"]
                                                  for r in subset) for name in BASE_NAMES}}
    per_codec={}
    for key in sorted({r["expected"] for r in rows}):
        subset=[r for r in rows if r["expected"]==key]
        per_codec[key]={"n":len(subset),**{name:sum(r["predictions"][name]["same_text"]
                                                  for r in subset) for name in BASE_NAMES}}
    return {"overall":metrics,"paired_vs_baseline":paired,"by_language":groups,"by_codec":per_codec}


def main(args):
    check_rev(args.fixtures,EXPECTED_CORPUS_REV)
    check_rev(args.training,EXPECTED_TRAIN_REV)
    if chardet.__version__!="7.7.0" or charset_normalizer.__version__!="3.5.2":
        raise RuntimeError("competitor version changed")
    train=[(raw,label,group) for raw,label,group,fold,digest in corpus(pathlib.Path(args.training))]
    cases,excluded,source_hash=cases_from_manifest(args.fixtures,args.manifest,train)
    residual,_=residual_subset(train)
    extras,gen=contrastive_augmentation(residual,per_direction=24,exclude_ebcdic=True)
    base=TrainableDetector.fit(residual,feature_profile="stable23")
    specialist=TrainableDetector.fit(residual+extras,feature_profile="stable23")
    routed=SelectiveLatinDetector(base,specialist)
    models={"baseline":base,"latin_only":specialist,"routed":routed}
    cases_out=[]
    for path,lang,codec,raw,text in cases:
        prediction={}
        for name,model in models.items():
            answer=decode(model,raw,bom_policy="preserve")
            prediction[name]={
                "encoding":answer.encoding,"same_text":answer.text==text,
                "exact_codec":(codecs.lookup(answer.encoding).name==codec
                               if answer.encoding else False),
                "abstained":answer.text is None,"ambiguous":answer.ambiguous,
            }
        for name in ("chardet7","charset_normalizer"):
            encoding,decoded=third_party(name,raw)
            try: same_label=bool(encoding and codecs.lookup(encoding).name==codec)
            except LookupError: same_label=False
            prediction[name]={"encoding":encoding,"same_text":decoded==text,
                              "exact_codec":same_label,"abstained":decoded is None}
        cases_out.append({"path":path,"lang":lang,"expected":codec,"sha256":sha(raw),
                          "bytes":len(raw),"predictions":prediction})
    result={
        "schema":"codeccat-archived-byte-external-one-shot-v1",
        "status":"CONSUMED_EXTERNAL; RESEARCH_ONLY",
        "source":"BYVoid/uchardet historical archived test fixtures",
        "source_revision":EXPECTED_CORPUS_REV,"locked_manifest_sha":source_hash,
        "source_freeze_run":"https://github.com/Johnny-Kao/CodecCat/actions/runs/41237703300",
        "source_limits":"archived codec-labelled test files, originally authored encoding not proven; test-only",
        "n_tested":len(cases_out),"excluded":excluded,
        "models":{
            "training_revision":EXPECTED_TRAIN_REV,
            "n_baseline_training":len(residual),
            "n_specialist_training":len(residual)+len(extras),
            "n_specialist_added":len(extras),
            "code_style":"actual two-model SelectiveLatinDetector with strict decoding",
        },
        "metric":"exact Unicode text equality; initial BOM PRESERVED; label accuracy secondary",
        "results":scores(cases_out),
        "limitations":[
            "Prior chardet test-data was used to train the compared research models",
            "This is one historical fixture suite, not representative real-world prevalence",
            "The original test bytes are not redistributed",
            "Unknown near-duplicate documents may remain",
            "Test code is held-out from training; further tuning on it invalidates fresh status",
        ],
    }
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    (out/"cases.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True)
                                         for x in cases_out)+"\n")
    print("INDEPENDENT_RAW_EVALUATION_BEGIN",flush=True)
    print(json.dumps({"n":result["n_tested"],"excluded":excluded,"results":result["results"]},
                     indent=2,sort_keys=True),flush=True)
    print("INDEPENDENT_RAW_EVALUATION_END",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fixtures",required=True)
    p.add_argument("--training",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",default="research-results/uchardet-one-shot")
    main(p.parse_args())
