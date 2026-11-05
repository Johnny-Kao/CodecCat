"""Single length-gated L2 research A/B on the frozen five mixed-source panels.

Inputs <=256 bytes use a separately trained short-prefix L2 model.
Inputs >256 bytes retain the identical baseline model and layered runtime.
No per-encoding rules, no package/runtime mutation, no manual test-case exceptions.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import json
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence
from five_panel_tournament import make_cases,distribute,strict,SEED

SHORT_LIMIT=256

class LengthRouted:
    def __init__(self,normal,short):
        self.normal=normal
        self.short=short
    def rank(self,data:bytes,*,final=True,structural_first=False):
        selected=self.short if len(data)<=SHORT_LIMIT else self.normal
        return selected.rank(data,final=final,structural_first=structural_first)

def fit_models(train):
    residual=residual_subset(train)[0]
    normal=fit(residual,"binary_high")
    # Keep the full-document observations, augment with two short windows.
    # Source groups stay in their original training fold. Drop ambiguous
    # same-byte windows if distinct original labels collapse to identical
    # prefixes; do not invent a ground truth for non-identifiable samples.
    raw_examples=[(raw,enc,group) for raw,enc,group in residual]
    for raw,enc,group in residual:
        if len(raw)>128: raw_examples.append((raw[:128],enc,group))
        if len(raw)>SHORT_LIMIT: raw_examples.append((raw[:SHORT_LIMIT],enc,group))
    label_sets={}
    for raw,enc,group in raw_examples:
        label_sets.setdefault(raw,set()).add(enc)
    short_train=[(raw,enc,group) for raw,enc,group in raw_examples if len(label_sets[raw])==1]
    short=fit(short_train,"binary_high")
    language=LanguageCoherence.fit(train)
    return normal,LengthRouted(normal,short),language

def add(stats,scope,base_ok,new_ok):
    c=stats[scope]
    c["n"]+=1
    c["base"]+=int(base_ok)
    c["routed"]+=int(new_ok)
    c["fixed"]+=int(not base_ok and new_ok)
    c["broken"]+=int(base_ok and not new_ok)

def summary(count):
    return {**dict(count),
            "base_pct":round(100*count["base"]/count["n"],3),
            "routed_pct":round(100*count["routed"]/count["n"],3)}

def main():
    rows=corpus(Path("training"))
    pool,_=make_cases(rows)
    panels=distribute(pool)
    trained={}
    for fold in (0,1,2):
        allowed=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
        trained[fold]=fit_models(allowed)
    trained[-1]=fit_models([(b,y,g) for b,y,g,k,_ in rows])
    overall=defaultdict(Counter)
    by_source=defaultdict(Counter)
    by_size=defaultdict(Counter)
    rounds=[]
    details=[]
    for number,panel in enumerate(panels,1):
        p=defaultdict(Counter)
        for record in panel:
            raw,enc=record["raw"],record["label"]
            expected=strict(raw,enc)
            assert expected is not None
            normal,routed,language=trained[record["fold"]]
            base=detect(normal,raw,language_model=language)
            test=detect(routed,raw,language_model=language)
            base_ok=bool(base.accepted and base.text==expected)
            new_ok=bool(test.accepted and test.text==expected)
            size="short<=256" if len(raw)<=SHORT_LIMIT else "long>256"
            for target,scope in ((overall,"all"),(by_source,record["source"]),
                                 (by_size,size),(p,"all"),(p,record["source"]),(p,size)):
                add(target,scope,base_ok,new_ok)
            details.append({"panel":number,"sha256":record["sha"],
                            "source":record["source"],"group":record["group"],
                            "bytes":len(raw),"truth":enc,
                            "base_encoding":base.encoding,"routed_encoding":test.encoding,
                            "base_correct":base_ok,"routed_correct":new_ok,
                            "changed":base.encoding!=test.encoding})
        rounds.append({"panel":number,**{name:summary(c) for name,c in p.items()}})
        print("LENGTH_BRANCH_PANEL_DONE",number,flush=True)
    assert len(details)==3092
    # Deterministic non-short routing: the same model, same language and same
    # input must produce the same output for all >256-byte inputs.
    assert all(x["base_encoding"]==x["routed_encoding"] and
               x["base_correct"]==x["routed_correct"]
               for x in details if x["bytes"]>SHORT_LIMIT)
    report={
        "status":"research-candidate-not-production",
        "seed":SEED,"cutoff_bytes":SHORT_LIMIT,
        "all":summary(overall["all"]),
        "by_source":{k:summary(v) for k,v in by_source.items()},
        "by_length":{k:summary(v) for k,v in by_size.items()},
        "panels":rounds,
        "limitations":[
            "Five panels are sourced from previously used fixture data; no independent confirmation.",
            "The short branch uses source-group-heldout originals plus 128/256-byte augmentations, excluding ambiguous same-byte prefixes.",
            "This evaluates one prespecified length cutoff; not tuned on the test cases.",
            "No changes to released models, main branch, or standard runtime."
        ]}
    out=Path("research-results/length-routed-ab")
    out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"cases.jsonl").write_text("\n".join(json.dumps(x) for x in details)+"\n")
    print("LENGTH_ROUTED_AB_RESULT",json.dumps(report),flush=True)

if __name__=="__main__":main()
