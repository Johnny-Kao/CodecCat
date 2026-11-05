"""Oracle ceiling audit for CodecCat current layered research baseline.

No tuning, model or runtime mutation. Evaluate the locked five-panel fixtures
using group-heldout folds, then inspect EVERY error's complete trained-class
score order with strict full-document decoding. Report:
- actual top-1 and layer attribution;
- top-k *oracle* coverage in score order and strict-valid score order;
- failure decomposition by length/source and L1/L3 paths;
- language-score best-candidate diagnostic (not a deployable oracle).

Oracle knows expected decoded text, so its numbers are UPPER BOUNDS only.
"""
from __future__ import annotations
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from five_panel_tournament import make_cases,distribute,strict,SEED
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence

DEPTHS=(1,2,3,5,8,16,32,64,999)
def add(c,entry):
    c["n"]+=1
    c["correct"]+=int(entry["correct"])
    if not entry["correct"]:
        c["errors"]+=1
        c["l2_top1_good"]+=int(entry["l2_top1_good"])
        c["top3_good"]+=int(entry["top3_good"])
        c["l2_intervention_harmed"]+=int(entry["l2_top1_good"])
        c["truth_codec_trained"]+=int(entry["truth_codec_trained"])
        if entry["best_valid_rank"] is None:c["not_in_all_valid"]+=1
        for depth in DEPTHS:
            c["oracle_valid_"+str(depth)]+=int(entry["best_valid_rank"] is not None and entry["best_valid_rank"]<=depth)
            c["oracle_raw_"+str(depth)]+=int(entry["best_raw_rank"] is not None and entry["best_raw_rank"]<=depth)

def main():
    rows=corpus(Path("training"))
    pool,_=make_cases(rows);panels=distribute(pool)
    models={}
    for fold in (0,1,2,-1):
        tr=[(b,y,g) for b,y,g,k,_ in rows if k!=fold] if fold!=-1 else [(b,y,g) for b,y,g,k,_ in rows]
        models[fold]=(fit(residual_subset(tr)[0],"binary_high"),LanguageCoherence.fit(tr))
        print("ORACLE_FOLD_TRAINED",fold,flush=True)
    totals=Counter();by_source=defaultdict(Counter)
    by_length=defaultdict(Counter);by_panel=defaultdict(Counter);by_reason=Counter()
    examples=[];errors=[]
    for idx,panel in enumerate(panels,1):
        for r in panel:
            raw=r["raw"];truth=strict(raw,r["label"])
            model,lang=models[r["fold"]]
            top=model.rank(raw,final=True)
            res=detect(model,raw,language_model=lang)
            correct=bool(res.accepted and res.text==truth)
            if correct:
                entry={"correct":True}
            else:
                l2_good=strict(raw,top.encoding)==truth if top.encoding else False
                top3_good=any(strict(raw,c)==truth for c in top.candidates)
                # FULL model score ordering, not shortened top3; include only
                # supported classes and strict full-document validity.
                x=model.vectorizer.transform([raw[:4096].decode("latin1")])
                clf=model.classifier
                scores=np.asarray(clf.decision_function(x)).reshape(-1)
                if len(clf.classes_)==2:scores=np.array([-scores[0],scores[0]])
                order=np.argsort(scores)[::-1]
                correct_in_raw=[];correct_in_valid=[]
                valid_count=0;correct_codecs=[]
                for raw_rank,j in enumerate(order,1):
                    enc=str(clf.classes_[int(j)])
                    text=strict(raw,enc)
                    if text is None:continue
                    valid_count+=1
                    if text==truth:
                        correct_in_raw.append(raw_rank)
                        correct_in_valid.append(valid_count)
                        correct_codecs.append(enc)
                if not correct_in_valid:reason="no-valid-equivalent-trained-codec"
                elif l2_good:reason="post-l2-overrode-correct"
                elif top3_good:reason="wrong-top3-ranking"
                elif min(correct_in_valid)<=16:reason="rank4-through-16"
                else:reason="deep-score-or-structural-forced"
                by_reason[reason]+=1
                entry={"correct":False,
                       "l2_top1_good":l2_good,"top3_good":top3_good,
                       "truth_codec_trained":r["label"] in set(str(k) for k in clf.classes_),
                       "best_raw_rank":min(correct_in_raw) if correct_in_raw else None,
                       "best_valid_rank":min(correct_in_valid) if correct_in_valid else None,
                       "correct_equivalents":correct_codecs[:12],
                       "full_valid_candidate_count":valid_count,
                       "reason":reason,
                       "source":r["source"],"group":r["group"],"label":r["label"],
                       "sha256":r["sha"],"bytes":len(raw),
                       "l2_choice":top.encoding,"runtime_choice":res.encoding,
                       "runtime_reason":res.reason,"l3_changed":res.language_changed,
                       "declared":res.declaration_used,
                       "l2_candidates":list(top.candidates)}
                errors.append(entry)
            add(totals,entry)
            add(by_source[r["source"]],entry)
            add(by_length["short<=256" if len(raw)<=256 else "long>256"],entry)
            add(by_panel[str(idx)],entry)
        print("ORACLE_PANEL_DONE",idx,flush=True)
    assert totals["n"]==3092 and totals["correct"]==2914,(totals["n"],totals["correct"])
    report={
        "seed":SEED,"training_rule":"group-heldout chardet folds; full chardet-only training for external cases",
        "correctness":"full strict decoded Unicode text equality, after leading BOM normalization",
        "totals":dict(totals),
        "by_source":{k:dict(v) for k,v in by_source.items()},
        "by_length":{k:dict(v) for k,v in by_length.items()},
        "panels":{k:dict(v) for k,v in by_panel.items()},
        "error_reasons":dict(by_reason),
        "limits":[
            "Oracle uses labeled truth to select rank; this cannot be implemented directly.",
            "A candidate's valid decoding is not evidence that its text is correct.",
            "Score order diagnostic excludes the L1 forced-decision priority.",
            "Existing development fixtures have influenced prior optimization; no blind confirmation.",
            "Some source fixture labels and original provenance remain independently unverified."
        ],
    }
    out=Path("research-results/oracle-ceiling-audit")
    out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (out/"errors.jsonl").write_text("\n".join(json.dumps(e,ensure_ascii=False) for e in errors)+"\n")
    print("ORACLE_AUDIT_RESULT",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
