"""Integrated acceptance diagnostics: source-group error taxonomy, model roundtrip, external comparison.

This is evidence collection, NOT a new candidate architecture nor blind release gate.
"""
from __future__ import annotations
import argparse,codecs,hashlib,json,sys,tempfile,time
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_training_strategy_audit import strict_text,family
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_gate import evaluate

def classify(raw,truth,truth_text,ans,possible):
    pred=str(ans.encoding) if ans.encoding is not None else None
    decoded=strict_text(raw,pred)
    if decoded is not None and decoded==truth_text:
        return "correct_text"
    if pred is None or decoded is None:
        return "abstained_or_undecodable"
    if truth not in possible:
        return "truth_structurally_excluded"
    if truth not in ans.candidates:
        return "truth_not_in_validated_top3"
    return "truth_in_top3_but_not_selected"

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--training",required=True)
 p.add_argument("--external",required=True)
 p.add_argument("--out",default="research-results/integrated-architecture-gate")
 a=p.parse_args()
 rows=corpus(Path(a.training))
 assert len(rows)==len({r[4] for r in rows})
 folds=[];cases=[];lifecycle=[]
 total_start=time.perf_counter()
 for fold in range(3):
    training=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
    held=[(b,y,g,s) for b,y,g,k,s in rows if k==fold]
    assert {g for _,_,g in training}.isdisjoint({g for _,_,g,_ in held})
    trimmed,_=residual_subset(training)
    start=time.perf_counter()
    model=TrainableDetector.fit(trimmed,feature_profile="stable23")
    fitting=time.perf_counter()-start
    # One real portable roundtrip for every fold: compare every held-out decision.
    with tempfile.TemporaryDirectory() as temp:
        file=Path(temp)/"model.zip"
        model.save_portable(file)
        restored=TrainableDetector.load_portable(file)
        assert restored.fingerprint==model.fingerprint
        model_bytes=file.stat().st_size
        counts=Counter()
        for raw,truth,group,digest in held:
            final=len(raw)<4096
            orig=model.rank(raw,final=final)
            again=restored.rank(raw,final=final)
            assert (orig.encoding,orig.candidates,orig.ambiguous)==(again.encoding,again.candidates,again.ambiguous),("roundtrip",digest)
            truth_text=strict_text(raw,truth)
            assert truth_text is not None
            gate=evaluate(raw,tuple(str(z) for z in model.classifier.classes_),final=final)
            category=classify(raw,truth,truth_text,orig,gate.possible)
            counts[category]+=1
            counts["n"]+=1
            if category!="correct_text":
                cases.append({"fold":fold,"sha256":digest,"group":group,"family":family(truth),
                "truth":truth,"predicted":str(orig.encoding) if orig.encoding else None,
                "top3":list(orig.candidates),"category":category,
                "truth_possible":truth in gate.possible,
                "truth_in_top3":truth in orig.candidates})
        folds.append({"fold":fold,"counts":dict(counts),"train":len(trimmed)})
        lifecycle.append({"fold":fold,"train_seconds":round(fitting,3),"portable_bytes":model_bytes,
                          "roundtrip_predictions_checked":len(held),"fingerprint":model.fingerprint})
    print("INTEGRATED_FOLD",fold,dict(counts),flush=True)
 # External benchmark: pinned comparator versions supplied via workflow
 import chardet,charset_normalizer
 ext=[];seen_text={hashlib.sha256(strict_text(b,y).replace("\r\n","\n").replace("\r","\n").lstrip("\ufeff").encode()).hexdigest() for b,y,_,_,_ in rows if strict_text(b,y) is not None}
 from external_accuracy_tournament import truth_label,measure,canonical
 finalmodel=TrainableDetector.fit(residual_subset([(b,y,g) for b,y,_,_,_ in rows])[0],feature_profile="stable23")
 for path in sorted((Path(a.external)/"test").rglob("*")):
    if not path.is_file():continue
    truth=truth_label(path)
    if truth is None:continue
    raw=path.read_bytes()
    ref=strict_text(raw,truth)
    if ref is None:continue
    semantic=hashlib.sha256(ref.replace("\r\n","\n").replace("\r","\n").lstrip("\ufeff").encode()).hexdigest()
    if semantic in seen_text:continue
    rank=finalmodel.rank(raw,final=len(raw)<4096)
    try:guess=chardet.detect(raw).get("encoding");ch=canonical(guess) if guess else None
    except Exception:ch=None
    try:best=charset_normalizer.from_bytes(raw).best();cn=canonical(best.encoding) if best else None
    except Exception:cn=None
    ext.append({"path":path.relative_to(a.external).as_posix(),"truth":truth,
                "correct":{"codeccat":measure(truth,raw,rank.encoding),
                           "chardet":measure(truth,raw,ch),
                           "charset_normalizer":measure(truth,raw,cn)}})
 totals=Counter()
 for fold in folds:totals.update(fold["counts"])
 tax=Counter(c["category"] for c in cases)
 report={"schema":"codeccat-integrated-architecture-v1",
 "status":"development-only-NOT-independent-release-evidence",
 "grouped_cv":{"n":totals["n"],"correct_text":totals["correct_text"],"error_count":len(cases),
               "error_taxonomy":dict(tax),"folds":folds},
 "model_lifecycle":{"roundtrip_checked":sum(x["roundtrip_predictions_checked"] for x in lifecycle),
                    "folds":lifecycle,"note":"Portable serialized model restored and gave identical decisions across all held-out cases."},
 "external_fixture":{"n":len(ext),"versions":{"chardet":chardet.__version__,"charset_normalizer":charset_normalizer.__version__},
                     "correct":{k:sum(x["correct"][k] is True for x in ext) for k in ("codeccat","chardet","charset_normalizer")}},
 "cautions":["No change to classifier or parameters; no architectural benefit asserted.",
              "Competitor-owned source and third-party fixtures share provenance; no blind confirmation.",
              "Semantic de-dup uses normalized decoded prefix, not near-duplicate detection.",
              "No time measurements for inference or process peak RAM.","Held-out group split uses existing corpus conventions."],
 "elapsed_seconds":round(time.perf_counter()-total_start,1)}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/"summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 (out/"errors.jsonl").write_text("\n".join(json.dumps(c,sort_keys=True) for c in cases)+"\n")
 (out/"external_cases.jsonl").write_text("\n".join(json.dumps(c,sort_keys=True) for c in ext)+"\n")
 print("INTEGRATED_ARCHITECTURE_RESULT_START",flush=True)
 print(json.dumps(report,indent=2),flush=True)
 print("INTEGRATED_ARCHITECTURE_RESULT_END",flush=True)
if __name__=="__main__":main()
