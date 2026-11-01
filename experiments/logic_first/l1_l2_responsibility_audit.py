"""Audit deterministic L1 vs learned L2 responsibilities without tuning."""
import argparse,collections,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_gate import evaluate
from vnext_training_strategy_audit import strict_text

def main():
 p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True);p.add_argument("--out",default="research-results/responsibility-audit");a=p.parse_args()
 rows=corpus(Path(a.fixtures));report=[];cases=[]
 for fold in range(3):
  train=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
  held=[(b,y,g,sha) for b,y,g,k,sha in rows if k==fold]
  trimmed,_=residual_subset(train);model=TrainableDetector.fit(trimmed,feature_profile="stable23")
  count=collections.Counter()
  for raw,truth,group,digest in held:
   final=len(raw)<4096
   evidence=evaluate(raw[:4096],tuple(str(c) for c in model.classifier.classes_),final=final)
   base=model.rank(raw,final=final)
   gated=model.rank(raw,final=final,structural_first=True)
   expected=strict_text(raw,truth)
   base_text=strict_text(raw,base.encoding)
   gated_text=strict_text(raw,gated.encoding)
   base_ok=base_text is not None and base_text==expected
   gated_ok=gated_text is not None and gated_text==expected
   truth_possible=truth in evidence.possible
   count["n"]+=1;count["truth_impossible"]+=int(not truth_possible)
   count["l1_unique"]+=int(len(evidence.possible)==1)
   count["l1_many"]+=int(len(evidence.possible)>1)
   count["l2_top3_miss"]+=int(truth not in base.candidates)
   count["base_correct_text"]+=base_ok;count["gated_correct_text"]+=gated_ok
   count["gating_help"]+=int(gated_ok and not base_ok)
   count["gating_harm"]+=int(base_ok and not gated_ok)
   count["gating_change"]+=int(gated.encoding!=base.encoding)
   if not base_ok or gated.encoding!=base.encoding or not truth_possible:
    cases.append({"sha":digest,"fold":fold,"group":group,"truth":truth,
      "truth_possible":truth_possible,"l1_candidate_count":len(evidence.possible),
      "base":base.encoding,"gated":gated.encoding,"base_top3":base.candidates,
      "base_text_correct":base_ok,"gated_text_correct":gated_ok})
  report.append({"fold":fold,**count})
  print("RESPONSIBILITY_FOLD_DONE",fold,dict(count),flush=True)
 totals={k:sum(x[k] for x in report) for k in report[0] if k!="fold"}
 result={"schema":"codeccat-responsibility-audit-v1","status":"development-diagnostic-not-blind",
 "totals":totals,"folds":report,"limitations":["Existing training-origin chardet fixtures, not independent evidence.",
 "Truth-impossible may reflect prefix, alias or labeling issues and requires review.",
 "structural_first is experimental existing runtime option, not a new production rule.",
 "Only compares strict text; not benchmark timing or memory.","No model parameter changes."]}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 (out/"cases.jsonl").write_text("\n".join(json.dumps(c,sort_keys=True,default=list) for c in cases)+"\n")
 print("RESPONSIBILITY_AUDIT_START");print(json.dumps(result,indent=2));print("RESPONSIBILITY_AUDIT_END")
if __name__=="__main__":main()
