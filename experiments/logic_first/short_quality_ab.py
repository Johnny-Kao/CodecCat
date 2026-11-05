"""Paired A/B for short input C1/control-character anomaly guard.
Folds and five disjoint panels match baseline; long path is unchanged.
"""
from collections import Counter,defaultdict
from pathlib import Path
import json
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from five_panel_tournament import make_cases,distribute,strict,SEED
from codeccat.vnext_l3_language import LanguageCoherence
from codeccat.vnext_layered_pipeline import detect
from short_quality_route import ShortQualityRanker,SHORT_LIMIT

def add(stats,key,old,new):
    c=stats[key];c["n"]+=1;c["old"]+=int(old);c["new"]+=int(new)
    c["fixed"]+=int(new and not old);c["broken"]+=int(old and not new)

def main():
    rows=corpus(Path("training"));pool,_=make_cases(rows);panels=distribute(pool)
    models={}
    for fold in (0,1,2,-1):
        training=[(b,y,g) for b,y,g,k,_ in rows if k!=fold] if fold!=-1 else [(b,y,g) for b,y,g,k,_ in rows]
        m=fit(residual_subset(training)[0],"binary_high")
        models[fold]=(m,ShortQualityRanker(m),LanguageCoherence.fit(training))
    overall=defaultdict(Counter);by_source=defaultdict(Counter);by_length=defaultdict(Counter)
    rounds=[];changes=[]
    for p,panel in enumerate(panels,1):
        c=defaultdict(Counter)
        for sample in panel:
            b=sample["raw"];expected=strict(b,sample["label"])
            original,variant,lang=models[sample["fold"]]
            a=detect(original,b,language_model=lang)
            v=detect(variant,b,language_model=lang)
            a_ok=bool(a.accepted and a.text==expected)
            v_ok=bool(v.accepted and v.text==expected)
            size="short<=256" if len(b)<=SHORT_LIMIT else "long>256"
            for group,key in ((overall,"all"),(by_source,sample["source"]),
                              (by_length,size),(c,"all"),(c,sample["source"])):
                add(group,key,a_ok,v_ok)
            if len(b)>SHORT_LIMIT:assert (a.encoding,a.text,a.accepted)==(v.encoding,v.text,v.accepted)
            if a.encoding!=v.encoding:
                changes.append({"sha256":sample["sha"],"source":sample["source"],"fold":sample["fold"],
                    "bytes":len(b),"old":a.encoding,"new":v.encoding,"fixed":v_ok and not a_ok,
                    "broken":a_ok and not v_ok})
        rounds.append({"panel":p,"counts":{k:dict(v) for k,v in c.items()}})
        print("QUALITY_GUARD_PANEL_DONE",p,flush=True)
    report={"seed":SEED,"all":dict(overall["all"]),"by_source":{k:dict(v) for k,v in by_source.items()},
        "by_length":{k:dict(v) for k,v in by_length.items()},"panels":rounds,
        "changed":len(changes),"changed_cases":changes,
        "constraints":"Only <=256 bytes; no explicit labels or competitor calls; unchanged long path.",
        "limitation":"Previously reused development fixtures; not independent blind holdout."}
    out=Path("research-results/short-quality-ab");out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SHORT_QUALITY_AB_RESULT",json.dumps({k:v for k,v in report.items() if k!="changed_cases"}),flush=True)
    print("SHORT_QUALITY_AB_CHANGES",json.dumps(changes),flush=True)
if __name__=="__main__":main()
