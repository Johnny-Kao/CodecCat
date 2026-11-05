"""Diagnose <=256-byte failures before changing routing or model training.

Uses frozen five panels and identical fold-trained baseline. No tuning.
Separates wrong L2 top candidate, unreachable top three, L3/declaration
overrides, rejection, and same-text ambiguous codec labels.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from five_panel_tournament import make_cases,distribute,strict,SEED
from codeccat.vnext_l3_language import LanguageCoherence
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_declarations import choose
from codeccat.vnext_binary_preflight import inspect_text

LIMIT=256

def main():
    rows=corpus(Path("training"))
    pool,_=make_cases(rows);panels=distribute(pool)
    trained={}
    for fold in (0,1,2,-1):
        train=[(b,y,g) for b,y,g,k,sha in rows if k!=fold] if fold!=-1 else [(b,y,g) for b,y,g,k,sha in rows]
        model=fit(residual_subset(train)[0],"binary_high")
        language=LanguageCoherence.fit(train)
        trained[fold]=(model,language)
    reasons=Counter();by_source=defaultdict(Counter);by_panel=defaultdict(Counter)
    rows_out=[]
    for pi,panel in enumerate(panels,1):
        for row in panel:
            raw,label,source=row["raw"],row["label"],row["source"]
            if len(raw)>LIMIT:continue
            expected=strict(raw,label)
            if expected is None:continue
            model,language=trained[row["fold"]]
            ranked=model.rank(raw,final=True)
            runtime=detect(model,raw,language_model=language)
            preflight=inspect_text(raw)
            declaration=choose(raw,ranked.encoding) if ranked.encoding else None
            l3=language.rerank(ranked,raw,final=True,max_l2_margin=.5,min_language_advantage=.15)
            l2_correct=strict(raw,ranked.encoding)==expected if ranked.encoding else False
            top3_ok=[codec for codec in ranked.candidates if strict(raw,codec)==expected]
            pipeline_ok=bool(runtime.accepted and runtime.text==expected)
            reason="correct"
            if not pipeline_ok:
                if not preflight.text_possible:reason="preflight-reject"
                elif not top3_ok:reason="truth-not-in-top3"
                elif not l2_correct:reason="l2-ranking-or-l3-miss"
                elif runtime.declaration_used:reason="declaration-overrode-correct-l2"
                elif runtime.language_changed:reason="language-overrode-correct-l2"
                elif not runtime.accepted:reason="runtime-decode-reject"
                else:reason="decode-contract-or-other"
            reasons[reason]+=1
            by_source[source][reason]+=1;by_panel[pi][reason]+=1
            rows_out.append({"sha":row["sha"],"source":source,"panel":pi,
                "group":row["group"],"truth":label,"bytes":len(raw),
                "baseline":ranked.encoding,"top3":list(ranked.candidates),
                "runtime":runtime.encoding,"runtime_reason":runtime.reason,
                "correct":pipeline_ok,"reason":reason,
                "top3_reachable":bool(top3_ok),"l2_correct":l2_correct,
                "language_changed":runtime.language_changed,
                "declaration_used":runtime.declaration_used,
                "ambiguous_same_text":len(top3_ok)>1,
                "margin":float(ranked.scores[0]-ranked.scores[1]) if len(ranked.scores)>1 else None})
        print("SHORT_DIAGNOSTIC_PANEL_DONE",pi,flush=True)
    assert len(rows_out)==188,len(rows_out)
    report={"seed":SEED,"limit":LIMIT,"total":len(rows_out),
            "reasons":dict(reasons),
            "by_source":{k:dict(v) for k,v in by_source.items()},
            "by_panel":{str(k):dict(v) for k,v in by_panel.items()},
            "top3_reachable":sum(r["top3_reachable"] for r in rows_out),
            "l2_correct":sum(r["l2_correct"] for r in rows_out),
            "runtime_correct":sum(r["correct"] for r in rows_out),
            "multiple_same_text_candidates":sum(r["ambiguous_same_text"] for r in rows_out),
            "note":"Diagnostic only; scores computed on previously used development fixtures, with complete strict decoding."}
    out=Path("research-results/short-error-attribution")
    out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows_out)+"\n")
    print("SHORT_ERROR_ATTRIBUTION_RESULT",json.dumps(report),flush=True)

if __name__=="__main__":main()
