"""Paired research A/B: shared-base L2 with short-only conservative fine-tuning.

Train the original binary12 L2 on group-heldout development data; deep-copy
its fitted linear classifier and update only on naturally short documents.
The fitted vectorizer and complete class vocabulary remain shared.
Do not alter long-input inference or add codec-specific rules.
"""
from collections import Counter,defaultdict
from pathlib import Path
import copy,json
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from five_panel_tournament import make_cases,distribute,strict,SEED
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence

SHORT_LIMIT=256
# Constant step, one pass: avoid turning a tiny natural short cohort into a new model.
FINETUNE_ETA=.0005

class SharedBaseShortAdapter:
    def __init__(self,base,short):
        self.base=base
        self.short=short
    def rank(self,data,*,final=True,structural_first=False):
        selected=self.short if len(data)<=SHORT_LIMIT else self.base
        return selected.rank(data,final=final,structural_first=structural_first)

def train(rows):
    residual=residual_subset(rows)[0]
    base=fit(residual,"binary_high")
    originals=[(b,y,g) for b,y,g in rows if len(b)<=SHORT_LIMIT]
    counts={"n":len(originals),"class_count":len(set(y for _,y,_ in originals)),
            "all_classes":len(base.classifier.classes_)}
    adapted=copy.deepcopy(base.classifier)
    # Do not refit vocabulary or use original text from held-out document groups.
    if originals:
        adapted.set_params(learning_rate="constant",eta0=FINETUNE_ETA)
        strings=[b.decode("latin1") for b,_,_ in originals]
        x=base.vectorizer.transform(strings)
        y=[label for _,label,_ in originals]
        adapted.partial_fit(x,y,classes=base.classifier.classes_)
    new=TrainableDetector(vectorizer=base.vectorizer,classifier=adapted,
                          fingerprint="shared-base-short-only-research")
    return base,SharedBaseShortAdapter(base,new),LanguageCoherence.fit(rows),counts

def push(c,old,new):
    c["n"]+=1;c["base"]+=int(old);c["candidate"]+=int(new)
    c["fixed"]+=int(new and not old);c["broken"]+=int(old and not new)

def main():
    rows=corpus(Path("training"));pool,_=make_cases(rows);panels=distribute(pool)
    trained={}
    for fold in (0,1,2,-1):
        train_rows=[(b,y,g) for b,y,g,k,_ in rows if k!=fold] if fold!=-1 else [(b,y,g) for b,y,g,k,_ in rows]
        trained[fold]=train(train_rows)
    overall=Counter();by_source=defaultdict(Counter);by_length=defaultdict(Counter)
    panel_results=[];changes=[]
    for panel_index,panel in enumerate(panels,1):
        local=Counter()
        for row in panel:
            raw,label=row["raw"],row["label"]
            truth=strict(raw,label)
            base,adapter,language,_=trained[row["fold"]]
            a=detect(base,raw,language_model=language)
            b=detect(adapter,raw,language_model=language)
            original=bool(a.accepted and a.text==truth)
            updated=bool(b.accepted and b.text==truth)
            for c in (overall,local,by_source[row["source"]],
                      by_length["short<=256" if len(raw)<=SHORT_LIMIT else "long>256"]):
                push(c,original,updated)
            if len(raw)>SHORT_LIMIT:
                assert (a.encoding,a.text,a.accepted)==(b.encoding,b.text,b.accepted)
            if a.encoding!=b.encoding or a.accepted!=b.accepted:
                changes.append({"panel":panel_index,"sha256":row["sha"],
                                "source":row["source"],"length":len(raw),
                                "old":a.encoding,"new":b.encoding,
                                "fixed":updated and not original,
                                "broken":original and not updated})
        panel_results.append({"panel":panel_index,**dict(local)})
        print("SHARED_SHORT_PANEL_DONE",panel_index,flush=True)
    report={"seed":SEED,"method":"shared fitted L2 weights and full vocabulary; one conservative partial_fit pass on naturally <=256-byte training samples",
            "eta":FINETUNE_ETA,"cutoff":SHORT_LIMIT,
            "total":dict(overall),
            "by_source":{k:dict(c) for k,c in by_source.items()},
            "by_length":{k:dict(c) for k,c in by_length.items()},
            "panels":panel_results,"training":{str(k):v[3] for k,v in trained.items()},
            "changed":len(changes),
            "limitations":["Reused development fixture families, not independent confirmation.",
                           "The long path is strictly unchanged.",
                           "Training uses grouped folds and excludes held-out source groups.",
                           "No production code or released model changes."]}
    out=Path("research-results/shared-short-adapter");out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"changes.jsonl").write_text("\n".join(json.dumps(x) for x in changes)+"\n")
    print("SHARED_SHORT_ADAPTER_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
