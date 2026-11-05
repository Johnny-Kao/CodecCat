"""Outer grouped-fold paired evaluation of inner-OOF learned candidate switcher.

Reports honest fixed five-panel development results, not fresh blind holdout.
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
from oof_pairwise_switch import fit_oof

class Combined:
    def __init__(self,lang,switch):self.lang=lang;self.switch=switch
    def rerank(self,ranked,raw,*,final=True,max_l2_margin=.5,min_language_advantage=.15):
        return self.switch.rerank(ranked,raw,final=final,
            max_l2_margin=max_l2_margin,min_language_advantage=min_language_advantage)

def tally(c,old,new):
    c["n"]+=1;c["baseline"]+=int(old);c["candidate"]+=int(new)
    c["fixes"]+=int(new and not old);c["breaks"]+=int(old and not new)

def main():
    rows=corpus(Path("training"));pool,_=make_cases(rows);panels=distribute(pool)
    models={}
    for fold in (0,1,2,-1):
        tr=[(b,y,g) for b,y,g,k,sha in rows if k!=fold] if fold!=-1 else [(b,y,g) for b,y,g,k,sha in rows]
        switch=fit_oof(tr,LanguageCoherence,strict)
        model=fit(residual_subset(tr)[0],"binary_high")
        lang=LanguageCoherence.fit(tr)
        models[fold]=(model,lang,Combined(lang,switch.wrap(lang)),switch.training_info)
        print("OOF_OUTER_TRAINED",fold,json.dumps(switch.training_info),flush=True)
    totals=Counter();by_source=defaultdict(Counter);by_length=defaultdict(Counter)
    rounds=[];changes=[]
    for idx,panel in enumerate(panels,1):
        c=Counter()
        for row in panel:
            raw=row["raw"];expected=strict(raw,row["label"])
            model,lang,combined,_=models[row["fold"]]
            a=detect(model,raw,language_model=lang)
            z=detect(model,raw,language_model=combined)
            old_ok=bool(a.accepted and a.text==expected)
            new_ok=bool(z.accepted and z.text==expected)
            for bucket in (totals,c,by_source[row["source"]],
                           by_length["short<=256" if len(raw)<=256 else "long>256"]):
                tally(bucket,old_ok,new_ok)
            if a.encoding!=z.encoding or a.accepted!=z.accepted:
                changes.append({"panel":idx,"sha256":row["sha"],
                    "source":row["source"],"bytes":len(raw),"truth":row["label"],
                    "baseline":a.encoding,"oof":z.encoding,"fix":new_ok and not old_ok,
                    "break":old_ok and not new_ok})
        rounds.append({"panel":idx,**dict(c)})
        print("OOF_PAIRWISE_PANEL_DONE",idx,flush=True)
    assert totals["n"]==3092 and totals["baseline"]==2914
    report={"seed":SEED,"gate":"P(switch)>0.80 and L2 margin<=0.50",
        "overall":dict(totals),
        "by_source":{k:dict(v) for k,v in by_source.items()},
        "by_length":{k:dict(v) for k,v in by_length.items()},
        "panels":rounds,
        "oof_train":{str(k):v[3] for k,v in models.items()},
        "changed":len(changes),
        "limitations":[
            "Evaluation fixture families repeatedly used in prior development, not untouched independent confirmation.",
            "Outer chardet folds grouped by source; inner metadata grouping for switcher uses group IDs only for partitioning.",
            "External data excluded from training. No inference based on source names or labels.",
            "Top-two gate preserving existing L3 decisions. No guarantee of >=95% stability on arbitrary data."
        ]}
    p=Path("research-results/oof-pairwise-switch");p.mkdir(parents=True,exist_ok=True)
    (p/"report.json").write_text(json.dumps(report,indent=2))
    (p/"changed.jsonl").write_text("\n".join(json.dumps(x) for x in changes)+"\n")
    print("OOF_PAIRWISE_AB_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
