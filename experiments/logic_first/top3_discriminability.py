"""Classify recoverable top3 errors by discriminative evidence, without tuning."""
import json,collections,statistics
from pathlib import Path
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from l2_variant_matrix import valid
def badness(s):
 if not s:return 0
 return sum((ord(c)<32 and c not in "\r\n\t") or (127<=ord(c)<160) for c in s)/len(s)
def main():
 rows=corpus(Path("training"));c=collections.Counter();gaps=[];examples=[]
 for fold in range(3):
  train=residual_subset([(b,y,g) for b,y,g,k,_ in rows if k!=fold])[0]
  model=fit(train,"binary_high")
  for b,y,g,k,sha in rows:
   if k!=fold:continue
   result=model.rank(b,final=len(b)<4096);expected=valid(b,y)
   if valid(b,result.encoding)==expected:continue
   c["incorrect"]+=1
   matches=[i for i,z in enumerate(result.candidates) if valid(b,z)==expected]
   if not matches:c["outside_top3"]+=1;continue
   c["recoverable"]+=1
   best=valid(b,result.encoding)
   right=valid(b,result.candidates[matches[0]])
   if best is not None and right is not None:
    if badness(best)>badness(right):c["control_penalty_direction_correct"]+=1
    elif badness(best)<badness(right):c["control_penalty_direction_wrong"]+=1
    else:c["control_penalty_tied"]+=1
    if len(best)!=len(right):c["different_decoded_length"]+=1
    else:c["same_decoded_length"]+=1
   if len(result.scores)>matches[0]:
    gaps.append(float(result.scores[0]-result.scores[matches[0]]))
   if len(examples)<16:examples.append({"truth":y,"pred":result.encoding,"correct_candidate_rank":matches[0]+1,"score_gap":float(result.scores[0]-result.scores[matches[0]]),"top_text_badness":badness(best),"correct_text_badness":badness(right)})
 report={"counts":dict(c),"gap_median":statistics.median(gaps) if gaps else None,"gap_p90":sorted(gaps)[int(.9*(len(gaps)-1))] if gaps else None,"examples":examples,"notes":"These metrics are descriptive; codepage labels/true text only for after-the-fact audit. Never use held-out truth to choose live result."}
 out=Path("research-results/top3-discriminability");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(report,indent=2))
 print("TOP3_DISCRIMINABILITY_RESULT",json.dumps({k:v for k,v in report.items() if k!="examples"}),flush=True)
if __name__=="__main__":main()
