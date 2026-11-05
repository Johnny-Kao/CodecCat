"""True length-partitioned training, not truncation augmentation.

Fold-specific model A: train on original <=256-byte documents only.
Fold-specific model B: train on original >256-byte documents only.
At inference route by len(raw) before L2. Keep identical L1/L3 runtime,
and evaluate against the locked, disjoint five-panel corpus.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import json
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from five_panel_tournament import make_cases,distribute,strict,SEED
from codeccat.vnext_l3_language import LanguageCoherence
from codeccat.vnext_layered_pipeline import detect

CUTOFF=256
class LengthSplitModel:
    def __init__(self,short,long):
        self.short=short
        self.long=long
    def rank(self,raw,*,final=True,structural_first=False):
        model=self.short if len(raw)<=CUTOFF else self.long
        return model.rank(raw,final=final,structural_first=structural_first)

def train(train_rows):
    short=[(b,y,g) for b,y,g in train_rows if len(b)<=CUTOFF]
    long=[(b,y,g) for b,y,g in train_rows if len(b)>CUTOFF]
    # Original length stratification. No expansion or cross-label training
    # leakage from short variants of long source documents.
    full=fit(residual_subset(train_rows)[0],"binary_high")
    small=fit(short,"binary_high") if len({y for b,y,g in short})>1 else None
    large=fit(residual_subset(long)[0],"binary_high") if len({y for b,y,g in long})>1 else None
    return full,LengthSplitModel(small or full,large or full),LanguageCoherence.fit(train_rows),{
        "short_train":len(short),"long_train":len(long),
        "short_classes":len({y for b,y,g in short}),
        "long_classes":len({y for b,y,g in long}),
        "short_fallback":small is None,"long_fallback":large is None,
    }

def bump(counter,old,new):
    counter["n"]+=1
    counter["baseline"]+=int(old)
    counter["split"]+=int(new)
    counter["fixed"]+=int(new and not old)
    counter["broken"]+=int(old and not new)

def main():
    rows=corpus(Path("training"))
    pool,_=make_cases(rows)
    panels=distribute(pool)
    models={}
    for fold in (0,1,2,-1):
        training=[(b,y,g) for b,y,g,k,_ in rows if k!=fold] if fold!=-1 else [(b,y,g) for b,y,g,k,_ in rows]
        models[fold]=train(training)
    overall=Counter();by_source=defaultdict(Counter)
    by_length=defaultdict(Counter);panel_reports=[];changed=[]
    for pi,panel in enumerate(panels,1):
        pc=Counter()
        for r in panel:
            raw=r["raw"];truth=strict(raw,r["label"])
            baseline,split,language,train_info=models[r["fold"]]
            old=detect(baseline,raw,language_model=language)
            new=detect(split,raw,language_model=language)
            old_ok=bool(old.accepted and old.text==truth)
            new_ok=bool(new.accepted and new.text==truth)
            for group in (overall,by_source[r["source"]],by_length["short<=256" if len(raw)<=CUTOFF else "long>256"],pc):
                bump(group,old_ok,new_ok)
            if old.encoding!=new.encoding or old.accepted!=new.accepted:
                changed.append({"sha":r["sha"],"source":r["source"],"panel":pi,
                    "bytes":len(raw),"truth":r["label"],"old":old.encoding,"split":new.encoding,
                    "fixed":new_ok and not old_ok,"broken":old_ok and not new_ok})
        panel_reports.append({"panel":pi,**dict(pc)})
        print("LENGTH_SPLIT_PANEL_DONE",pi,flush=True)
    report={"seed":SEED,"cutoff":CUTOFF,"total":dict(overall),
        "by_source":{k:dict(c) for k,c in by_source.items()},
        "by_length":{k:dict(c) for k,c in by_length.items()},
        "panels":panel_reports,
        "training":{str(k):v[3] for k,v in models.items()},
        "changed":len(changed),
        "note":"No document truncation or augmentation. Models trained on separate original-length strata, source-group-heldout. Development fixtures only, not blind confirmation."}
    out=Path("research-results/true-length-split-ab");out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"changed.jsonl").write_text("\n".join(json.dumps(x) for x in changed)+"\n")
    print("TRUE_LENGTH_SPLIT_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
