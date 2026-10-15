"""Residual training experiment: stable structural decisions handled by L1.

Compares baseline and residual-focused L2 on EXACT same source-group folds.
No R21. No release approval inferred from competitor-owned data.
"""
import argparse,collections,hashlib,json,time
from pathlib import Path
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_bom import inspect_bom
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus

def stable_structural_guess(data,known):
    """Apply only deterministic evidence already understood by the actual runtime."""
    sig=inspect_bom(data,final=len(data)<4096)
    if sig.incomplete:return None
    if sig.encoding:
        family="-".join(sig.encoding.split("-")[:2])
        prefer=family if family in ("utf-16","utf-32") and family in known else sig.encoding
        if prefer in known:return prefer
        return None
    if len(data)<32:return None
    pairs=min(len(data),4096)//2
    even=data[:pairs*2:2].count(0);odd=data[1:pairs*2:2].count(0)
    if max(even,odd)>=max(12,int(pairs*.30)) and max(even,odd)>=4*max(1,min(even,odd)):
        result="utf-16-be" if even>odd else "utf-16-le"
        if result in known:return result
    return None

def residual_subset(training,min_anchors=2):
    """Drop redundant L1-solved samples only, with per-label coverage anchors.

    Does not use the ground truth to make inference decisions. During training,
    the ground truth determines whether an apparent L1 decision is safe to remove
    from L2. A mistaken L1 prediction remains in training as a hard negative.
    """
    labels={label for _,label,_ in training}
    per_label=collections.Counter(label for _,label,_ in training)
    covered=collections.Counter();chosen=[];drop=collections.Counter()
    for data,label,group in training:
        decision=stable_structural_guess(data,labels)
        if decision==label and per_label[label]-covered[label]>min_anchors:
            covered[label]+=1
            drop[label]+=1
        else:
            chosen.append((data,label,group))
    assert {label for _,label,_ in chosen}==labels
    return chosen,dict(drop)

def run(args):
    rows=corpus(Path(args.fixtures))
    reports=[];paired=[]
    for fold in range(3):
        train=[(b,label,g) for b,label,g,k,_ in rows if k!=fold]
        test=[(b,label,g,sha) for b,label,g,k,sha in rows if k==fold]
        assert set(x[2] for x in train).isdisjoint(set(x[2] for x in test))
        trimmed,dropped=residual_subset(train)
        baseline=TrainableDetector.fit(train)
        focused=TrainableDetector.fit(trimmed)
        ctr=collections.Counter()
        for data,truth,g,sha in test:
            baseline_pred=baseline.rank(data,final=len(data)<4096).encoding
            focused_pred=focused.rank(data,final=len(data)<4096).encoding
            ctr["n"]+=1
            ctr["baseline"]+=baseline_pred==truth
            ctr["residual"]+=focused_pred==truth
            ctr["help"]+=focused_pred==truth and baseline_pred!=truth
            ctr["harm"]+=focused_pred!=truth and baseline_pred==truth
            paired.append({"sha":sha,"fold":fold,"truth":truth,"baseline":baseline_pred,"residual":focused_pred})
        report={"fold":fold,"training":len(train),"residual_training":len(trimmed),"dropped":len(train)-len(trimmed),
                "dropped_by_label":dropped,**dict(ctr)}
        reports.append(report)
        print("RESIDUAL_FOLD_DONE",fold,flush=True)
    totals={name:sum(x[name] for x in reports) for name in ("n","baseline","residual","help","harm","dropped")}
    totals["baseline_pct"]=round(100*totals["baseline"]/totals["n"],3)
    totals["residual_pct"]=round(100*totals["residual"]/totals["n"],3)
    result={"source":"chardet/test-data: competitor-owned development ONLY","schema":"vnext-residual-training-study-1","totals":totals,"folds":reports,
      "method":"same model, same features, same grouped folds; only L1-resolved training rows removed; keeps >=2 per class"}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(result,indent=2))
    (out/"paired.jsonl").write_text("\n".join(map(json.dumps,paired))+"\n")
    print("RESIDUAL_RESULT_BEGIN");print(json.dumps(result,indent=2));print("RESIDUAL_RESULT_END")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True)
    p.add_argument("--output",default="research-results/vnext-residual-study");run(p.parse_args())
