"""One-time score of separately FROZEN Git translation corpus.

The Git source manifest was frozen *before this scoring script existed*:
https://github.com/Johnny-Kao/CodecCat/actions/runs/41239218591

No Git bytes enter training, no payload bytes are uploaded as artifacts.
All model candidates are unchanged from their earlier declaration.
"""
from __future__ import annotations
import argparse,codecs,collections,json,pathlib,subprocess,sys
import chardet,charset_normalizer
from charset_normalizer import from_bytes
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_decoding import decode
from codeccat.vnext_selective import SelectiveLatinDetector
from codeccat.vnext_utf8_evidence import Utf8EvidenceDetector

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from vnext_vim_catalog_freeze import catalog_samples,fragments,take_window,sha,HEADER,POSITIONS,WINDOW
from vnext_git_catalog_freeze import GIT_REV
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_content_safe_contrast import contrastive_augmentation
from vnext_training_strategy_audit import strict_text

SOURCE_MANIFEST_SHA="2b966b9a344792cd3cc9aa8df14d7bbc815f8078061f14b0cf121f0b9057a3d2"
TRAIN_SHA="b0c0d206913ba118ffea2f7d306883e7a4a76db7"
METHODS=("baseline","latin_specialist","selective","utf8_rescue","combined","chardet7","charset_normalizer")


def check_rev(repo,sha_expected):
    found=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if found!=sha_expected:raise RuntimeError(f"unexpected repo commit: {found}")


def sample_bytes(filepath,codec,window):
    payload=fragments(filepath.read_bytes())
    offset=round((len(payload)-WINDOW)*POSITIONS[window])
    return take_window(payload,max(0,offset),codec)


def frozen_cases(root,manifest):
    j=json.loads(pathlib.Path(manifest).read_text())
    if j["sha256"]!=SOURCE_MANIFEST_SHA or j["revision"]!=GIT_REV:
        raise RuntimeError("unrecognized frozen source; do not score")
    fresh=[]
    for path in sorted((pathlib.Path(root)/"po").glob("*.po")):
        rows,rec=catalog_samples(path)
        registered=next((x for x in j["catalogs"] if x["path"]==path.name),None)
        if not registered or rec!=registered:
            raise RuntimeError(f"source file changed since preregistration: {path.name}")
        fresh+=rows
    if fresh!=j["windows"]:
        raise RuntimeError("unfrozen sample extraction changed")
    result=[]
    for item in fresh:
        file=pathlib.Path(root)/"po"/item["file"]
        raw=sample_bytes(file,item["codec"],item["window"])
        if raw is None or sha(raw)!=item["sha256"]:
            raise RuntimeError("raw source bytes mismatch frozen manifest")
        text=raw.decode(item["codec"],"strict")
        if sha(text.encode("utf-8","surrogatepass"))!=item["decoded_sha256"]:
            raise RuntimeError("decoded reference text mismatch")
        result.append({"raw":raw,"text":text,"file":item["file"],
                       "label":item["codec"],"window":item["window"],"sha256":item["sha256"]})
    return result


def thirdparty(name,data):
    try:
        if name=="chardet7":
            codec=chardet.detect(data).get("encoding")
        else:
            m=from_bytes(data).best()
            codec=m.encoding if m is not None else None
        return codec,data.decode(codec,"strict") if codec is not None else None
    except (UnicodeError,LookupError,TypeError,ValueError):
        return None,None


def counters(rows):
    n=len(rows)
    ret={"n":n,"overall":{},"paired":{},"by_file":{},"by_codec":{}}
    for name in METHODS:
        preds=[r["predictions"][name] for r in rows]
        ret["overall"][name]={"text_correct":sum(r["text_correct"] for r in preds),
                              "label_correct":sum(r["label_correct"] for r in preds),
                              "abstained":sum(r["abstained"] for r in preds),"n":n}
    for name in METHODS[1:]:
        h=sum(r["predictions"][name]["text_correct"] and
              not r["predictions"]["baseline"]["text_correct"] for r in rows)
        d=sum(r["predictions"]["baseline"]["text_correct"] and
              not r["predictions"][name]["text_correct"] for r in rows)
        ret["paired"][name]={"helps":h,"harms":d,"net":h-d}
    for key,column in (("by_file","file"),("by_codec","expected_codec")):
        for group in sorted({r[column] for r in rows}):
            subset=[r for r in rows if r[column]==group]
            ret[key][group]={"n":len(subset),**{
                model:sum(r["predictions"][model]["text_correct"] for r in subset)
                for model in METHODS}}
    return ret


def run(args):
    check_rev(args.source,GIT_REV);check_rev(args.training,TRAIN_SHA)
    if chardet.__version__!="7.7.0" or charset_normalizer.__version__!="3.5.2":
        raise RuntimeError("unfrozen competitor versions")
    cases=frozen_cases(args.source,args.manifest)
    train=[(b,label,group) for b,label,group,fold,digest in corpus(pathlib.Path(args.training))]
    texts={sha(strict_text(raw,label).encode("utf-8","surrogatepass"))
           for raw,label,group in train}
    seen=set();valid=[];drops=collections.Counter()
    for case in cases:
        t=sha(case["text"].encode("utf-8","surrogatepass"))
        if t in texts:
            drops["training_text_overlap"]+=1;continue
        if case["sha256"] in seen:
            drops["duplicate_raw_window"]+=1;continue
        seen.add(case["sha256"]);valid.append(case)
    if len(valid)<65:
        raise RuntimeError(f"too few uncontaminated independent-source byte windows: {len(valid)}")
    baseline_samples,_=residual_subset(train)
    extras,augmentation=contrastive_augmentation(
        baseline_samples,per_direction=24,exclude_ebcdic=True)
    original=TrainableDetector.fit(baseline_samples,feature_profile="stable23")
    latin=TrainableDetector.fit(baseline_samples+extras,feature_profile="stable23")
    select=SelectiveLatinDetector(original,latin)
    models={
        "baseline":original,
        "latin_specialist":latin,
        "selective":select,
        "utf8_rescue":Utf8EvidenceDetector(original,min_codepoints=16),
        "combined":Utf8EvidenceDetector(select,min_codepoints=16),
    }
    output=[]
    for case in valid:
        predictions={}
        for name,model in models.items():
            p=decode(model,case["raw"],bom_policy="preserve")
            try:identical_codec=bool(p.encoding and
                                    codecs.lookup(p.encoding).name==case["label"])
            except LookupError: identical_codec=False
            predictions[name]={"label":p.encoding,"text_correct":p.text==case["text"],
                               "label_correct":identical_codec,
                               "abstained":p.text is None,"ambiguous":p.ambiguous}
        for name in ("chardet7","charset_normalizer"):
            label,decoded=thirdparty(name,case["raw"])
            try:identical_codec=bool(label and codecs.lookup(label).name==case["label"])
            except LookupError:identical_codec=False
            predictions[name]={"label":label,"text_correct":decoded==case["text"],
                               "label_correct":identical_codec,
                               "abstained":decoded is None}
        output.append({
            "file":case["file"],"window":case["window"],"expected_codec":case["label"],
            "sha256":case["sha256"],"predictions":predictions,
        })
    report={
        "schema":"codeccat-frozen-independent-git-byte-confirmation-v1",
        "source_repo":"git/git","source_rev":GIT_REV,
        "frozen_before_scoring":"https://github.com/Johnny-Kao/CodecCat/actions/runs/41239218591",
        "frozen_manifest_digest":SOURCE_MANIFEST_SHA,
        "source_role":"UNSEEN_EXTERNAL_CONFIRMATION_CONSUMED_ONCE",
        "metric":"strict identical decoded Unicode; original source headers stripped; label separate",
        "source_catalogs":len({r["file"] for r in valid}),
        "n_frozen":len(cases),"n_valid":len(valid),"excluded":dict(drops),
        "training":"chardet/test-data pinned; source/file and decoded content disjoint",
        "result":counters(output),
        "caveats":[
            "Four windows of one Git PO are correlated, not independent files",
            "Most Git catalogs are UTF-8, not representative encoding prevalence",
            "Some PO snippets contain original gettext escaping bytes",
            "Git license applies; raw source bytes are not exported or used to fit",
            "After this run the Git PO suite is consumed development evidence, not new blind evidence",
        ],
    }
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    (out/"paired.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True)
                                          for x in output)+"\n")
    print("GIT_PO_CONFIRMATION_BEGIN")
    print(json.dumps({x:report[x] for x in ("source_catalogs","n_frozen","n_valid","excluded","result")},
                     sort_keys=True,indent=2))
    print("GIT_PO_CONFIRMATION_END")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--training",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",default="research-results/git-po-confirmation")
    run(p.parse_args())
