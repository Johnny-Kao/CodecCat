"""Training-fit audit BEFORE further generalization or parameter tuning.

Separates memorization, unseen source groups, indistinguishable labels, and
actual decoded-text errors. A high train fit is NOT a generalization metric.
"""
import argparse,collections,hashlib,json,sys
from pathlib import Path
from codeccat.vnext_strong_l2 import TrainableDetector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset,stable_structural_guess

def score(model,rows):
    counts=collections.Counter();failures=collections.Counter();examples=[]
    for raw,truth,group,fold,sha in rows:
        prediction=model.rank(raw,final=len(raw)<4096).encoding
        try:same_text=prediction is not None and raw.decode(truth,"strict")==raw.decode(prediction,"strict")
        except (UnicodeError,LookupError):same_text=False
        counts["n"]+=1
        counts["exact"]+=int(prediction==truth)
        counts["same_text"]+=int(same_text)
        counts["different_text"]+=int(not same_text)
        counts["label_only"]+=int(same_text and prediction!=truth)
        counts["unknown"]+=int(prediction is None)
        counts["structural_correct"]+=int(stable_structural_guess(raw,set(model.classifier.classes_))==truth)
        if not same_text:
            failures[(truth,prediction)]+=1
            if len(examples)<25:examples.append(dict(truth=truth,prediction=prediction,group=group,sha=sha))
    return dict(counts,top_actual_text_errors=[dict(truth=k[0],pred=k[1],n=v) for k,v in failures.most_common(30)],examples=examples)

def run(a):
    rows=corpus(Path(a.fixtures))
    train=[(b,y,g) for b,y,g,k,sha in rows]
    trimmed,_=residual_subset(train)
    models={"all":TrainableDetector.fit(train),"residual_capped":TrainableDetector.fit(trimmed)}
    report={"n_unique":len(rows),"n_source_groups":len({r[2] for r in rows}),
            "note":"in-sample FIT ONLY; cannot be used for release accuracy, generalization or holdout tuning",
            "train_fit":{name:score(m,rows) for name,m in models.items()}}
    for name,m in models.items():
        print("FIT_AUDIT_MODEL_DONE",name,flush=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("FIT_AUDIT_BEGIN");print(json.dumps(report,indent=2));print("FIT_AUDIT_END")
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--fixtures",required=True)
    parser.add_argument("--output",default="research-results/vnext-fit-audit")
    run(parser.parse_args())
