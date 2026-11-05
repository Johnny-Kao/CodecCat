"""Fixed five-panel A/B: baseline L3 vs L3 + conditional byte-bigram.

No new trained encoding classes; same L2 and source-group-heldout folds.
Only previously ambiguous top-three decisions can change.
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
from conditional_byte_bigram import ConditionalByteBigram

class L3WithByteEvidence:
    def __init__(self,language,byte):
        self.language=language;self.byte=byte
    def rerank(self,ranked,raw,*,final=True,max_l2_margin=.5,min_language_advantage=.15):
        original=self.language.rerank(ranked,raw,final=final,
             max_l2_margin=max_l2_margin,min_language_advantage=min_language_advantage)
        if original.changed or original.encoding is None:return original
        candidate=self.byte.rerank(ranked,raw,final=final,
             max_l2_margin=max_l2_margin,min_language_advantage=min_language_advantage)
        return candidate if candidate.changed else original

def incr(c,old,new):
    c["n"]+=1;c["base"]+=int(old);c["candidate"]+=int(new)
    c["fixed"]+=int(new and not old);c["broken"]+=int(old and not new)

def main():
    rows=corpus(Path("training"));pool,_=make_cases(rows);panels=distribute(pool)
    models={}
    for fold in (0,1,2,-1):
        train=[(b,y,g) for b,y,g,k,sha in rows if k!=fold] if fold!=-1 else [(b,y,g) for b,y,g,k,sha in rows]
        l2=fit(residual_subset(train)[0],"binary_high")
        lang=LanguageCoherence.fit(train)
        evidence=ConditionalByteBigram.fit(train)
        models[fold]=(l2,lang,L3WithByteEvidence(lang,evidence))
        print("BYTE_BIGRAM_FOLD_TRAINED",fold,flush=True)
    total=Counter();by_source=defaultdict(Counter);by_length=defaultdict(Counter);rounds=[];changes=[]
    for idx,panel in enumerate(panels,1):
        per_round=Counter()
        for record in panel:
            raw=record["raw"];expected=strict(raw,record["label"])
            model,lang,combined=models[record["fold"]]
            a=detect(model,raw,language_model=lang)
            z=detect(model,raw,language_model=combined)
            good=bool(a.accepted and a.text==expected)
            better=bool(z.accepted and z.text==expected)
            for c in (total,by_source[record["source"]],per_round,
                       by_length["short<=256" if len(raw)<=256 else "long>256"]):
                incr(c,good,better)
            if a.encoding!=z.encoding or a.accepted!=z.accepted:
                changes.append({"sha256":record["sha"],"panel":idx,
                    "source":record["source"],"truth":record["label"],
                    "bytes":len(raw),"baseline":a.encoding,"candidate":z.encoding,
                    "fixed":better and not good,"broken":good and not better})
        rounds.append({"panel":idx,**dict(per_round)})
        print("BYTE_BIGRAM_PANEL_DONE",idx,flush=True)
    assert total["n"]==3092 and total["base"]==2914
    report={"seed":SEED,"overall":dict(total),
        "by_source":{k:dict(v) for k,v in by_source.items()},
        "by_length":{k:dict(v) for k,v in by_length.items()},
        "panels":rounds,"changed":len(changes),
        "method":"trained per-codec byte-bigram log likelihood ratio; top3 only; L2 margin <=0.5; independent advantage >0.25; preserve existing L3 changes",
        "limits":["Previously reused source fixtures, not blind testing.",
                  "Only source-group-heldout training for chardet rows.",
                  "No production code modifications or tuning on individual cases."]}
    out=Path("research-results/conditional-bigram-ab");out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"changes.jsonl").write_text("\n".join(json.dumps(c) for c in changes)+"\n")
    print("CONDITIONAL_BIGRAM_AB_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
