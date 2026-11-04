"""Paired v1/v2 evaluation: identical five panels and frozen fold split.

V2 only swaps opt-in language reranker, never touches shipped Alpha/main.
"""
from collections import Counter,defaultdict
from pathlib import Path
import json
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence
from candidate_quality_reranker import QualityReranker
from five_panel_tournament import make_cases,distribute,strict,SEED

def main():
    rows=corpus(Path("training"))
    pool,_=make_cases(rows)
    panels=distribute(pool)
    folds={}
    for fold in range(3):
        train=[(b,y,g) for b,y,g,k,sha in rows if k!=fold]
        model=fit(residual_subset(train)[0],"binary_high")
        language=LanguageCoherence.fit(train)
        folds[fold]=(model,language,QualityReranker(language))
    train=[(b,y,g) for b,y,g,k,sha in rows]
    model=fit(residual_subset(train)[0],"binary_high")
    language=LanguageCoherence.fit(train)
    outside=(model,language,QualityReranker(language))
    stats=defaultdict(lambda:defaultdict(Counter))
    per_panel=[]
    cases=[]
    for idx,panel in enumerate(panels):
        round_counts=defaultdict(Counter)
        for r in panel:
            b,label,source=r["raw"],r["label"],r["source"]
            expected=strict(b,label)
            m,lang,new=folds[r["fold"]] if source=="chardet" else outside
            a=detect(m,b,language_model=lang)
            z=detect(m,b,language_model=new)
            a_ok=a.accepted and a.text==expected
            z_ok=z.accepted and z.text==expected
            for scope in ("all",source):
                ctr=stats[scope]["total"]
                ctr["n"]+=1;ctr["base"]+=int(a_ok);ctr["v2"]+=int(z_ok)
                ctr["fixed"]+=int(z_ok and not a_ok);ctr["broken"]+=int(a_ok and not z_ok)
            c=round_counts[source]
            c["n"]+=1;c["base"]+=int(a_ok);c["v2"]+=int(z_ok)
            c["fixed"]+=int(z_ok and not a_ok);c["broken"]+=int(a_ok and not z_ok)
            cases.append({"panel":idx+1,"sha":r["sha"],"source":source,"truth":label,
                "base":a.encoding,"v2":z.encoding,"base_ok":a_ok,"v2_ok":z_ok,
                "changed":a.encoding!=z.encoding,"reason":z.reason})
        per_panel.append({"panel":idx+1,"sources":{k:dict(v) for k,v in round_counts.items()},
                          "overall":{k:sum(v[k] for v in round_counts.values())
                                      for k in ("n","base","v2","fixed","broken")}})
        print("V2_AB_FOLD_DONE",idx+1,flush=True)
    report={"seed":SEED,"status":"research-opt-in-not-production",
        "totals":{scope:dict(details["total"]) for scope,details in stats.items()},
        "panels":per_panel,
        "caveat":"All three source fixture families have already influenced iterative model design. This is development comparison, not independent blind-test success."}
    p=Path("research-results/candidate-v2-ab");p.mkdir(parents=True,exist_ok=True)
    (p/"report.json").write_text(json.dumps(report,indent=2))
    (p/"cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases)+"\n")
    print("V2_AB_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
