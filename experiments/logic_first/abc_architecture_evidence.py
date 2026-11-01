"""Predeclared A/B/C evidence experiment. No SGD hyperparameter search.
B uses existing structural-first contract; C uses existing selective L3.
This is development comparison, not a promotion gate.
"""
import argparse,json,sys,time
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_training_strategy_audit import strict_text
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_l3_language import LanguageCoherence

def main():
 p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True);p.add_argument("--out",default="research-results/abc-architecture");a=p.parse_args()
 rows=corpus(Path(a.fixtures));cases=[];folds=[]
 for fold in range(3):
  train=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
  held=[(b,y,g,s) for b,y,g,k,s in rows if k==fold]
  assert {g for _,_,g in train}.isdisjoint({g for _,_,g,_ in held})
  trimmed,_=residual_subset(train)
  model=TrainableDetector.fit(trimmed,feature_profile="stable23")
  language=LanguageCoherence.fit(trimmed)
  totals=Counter()
  for raw,truth,group,digest in held:
   final=len(raw)<4096
   a_ans=model.rank(raw,final=final)
   b_ans=model.rank(raw,final=final,structural_first=True)
   c_ans=language.rerank(b_ans,raw,final=final)
   predictions={"A":a_ans.encoding,"B":b_ans.encoding,"C":c_ans.encoding}
   expected=strict_text(raw,truth)
   assert expected is not None
   results={name:strict_text(raw,pred)==expected if pred else False for name,pred in predictions.items()}
   totals["n"]+=1
   for name in predictions:
    totals[name+"_correct"]+=int(results[name])
    totals[name+"_abstain"]+=int(predictions[name] is None)
   totals["l3_called"]+=int(c_ans.called);totals["l3_changed"]+=int(c_ans.changed)
   totals["C_help"]+=int(results["C"] and not results["A"])
   totals["C_harm"]+=int(results["A"] and not results["C"])
   if c_ans.changed or not results["A"]:
    cases.append({"sha256":digest,"group":group,"fold":fold,"truth":truth,
     "predictions":predictions,"correct":results,"l3_reason":c_ans.reason})
  folds.append({"fold":fold,**totals})
  print("ABC_FOLD_COMPLETE",fold,dict(totals),flush=True)
 totals=Counter()
 for f in folds:totals.update({k:v for k,v in f.items() if k!="fold"})
 report={"schema":"codeccat-abc-architecture-v1","status":"development-only",
 "implementation":{"A":"stable23 residual current rank","B":"same rank with existing structural_first=True (no novel structural evidence)","C":"B plus pretrained selective LanguageCoherence L3 with existing default gates"},
 "totals":dict(totals),"folds":folds,
 "limitations":["B is a structural-first control, NOT newly implemented CJK evidence.",
 "C relies on language model trained on same development training partition.",
 "No new independently certified source corpus or speed/RAM measurement.",
 "No model parameter tuning, no alpha changes."]}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 (out/"cases.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True) for x in cases)+"\n")
 print("ABC_REPORT_START",flush=True);print(json.dumps(report,indent=2),flush=True);print("ABC_REPORT_END",flush=True)
if __name__=="__main__":main()
