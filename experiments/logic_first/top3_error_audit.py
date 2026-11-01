"""Top-3 score/margin audit for frozen grouped-fold benchmark; no tuning."""
import json,collections,sys
from pathlib import Path
from mixed_random_benchmark import build,decode
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
def main():
 rows,selected=build();byfold=collections.defaultdict(list)
 for r in selected:
  if r["source"]=="chardet":byfold[r["fold"]].append(r)
 result=[];stats=collections.defaultdict(collections.Counter)
 for fold in range(3):
  train=residual_subset([(b,y,g) for b,y,g,k,_ in rows if k!=fold])[0]
  model=TrainableDetector.fit(train,feature_profile="stable23")
  for r in byfold[fold]:
   ranked=model.rank(r["raw"],final=len(r["raw"])<4096)
   correct=decode(r["raw"],ranked.encoding)==decode(r["raw"],r["label"])
   true_decoded=decode(r["raw"],r["label"])
   true_rank=next((i+1 for i,c in enumerate(ranked.candidates) if decode(r["raw"],c)==true_decoded),None)
   gap=(ranked.scores[0]-ranked.scores[1]) if len(ranked.scores)>1 else None
   key=r["label"];s=stats[key];s["n"]+=1
   if not correct:
    s["errors"]+=1;s["wrong_top3_recoverable"]+=int(true_rank is not None)
    if true_rank is None:s["truth_absent"]+=1
   result.append({"sha":r["sha"],"truth":key,"prediction":ranked.encoding,"correct":correct,
     "truth_rank":true_rank,"top3":ranked.candidates,"scores":ranked.scores,"score_margin":gap})
  print("TOP3_FOLD_DONE",fold,flush=True)
 errors=[r for r in result if not r["correct"]]
 margins=sorted(r["score_margin"] for r in errors if r["score_margin"] is not None)
 byrank=collections.Counter(r["truth_rank"] if r["truth_rank"] is not None else "absent" for r in errors)
 summary={"n":len(result),"errors":len(errors),"error_truth_rank_distribution":{str(k):v for k,v in byrank.items()},
  "error_margin_p50":margins[len(margins)//2] if margins else None,
  "error_margin_p90":margins[int(.9*(len(margins)-1))] if margins else None,
  "recoverable_top3":sum(r["truth_rank"] is not None for r in errors),
  "truth_absent_top3":sum(r["truth_rank"] is None for r in errors),
  "error_by_family":[{"encoding":k,**dict(v)} for k,v in sorted(stats.items(),key=lambda kv:-kv[1]["errors"]) if v["errors"]],
  "examples":errors[:30]}
 out=Path("research-results/top3-audit");out.mkdir(parents=True,exist_ok=True)
 (out/"summary.json").write_text(json.dumps(summary,indent=2))
 (out/"cases.jsonl").write_text("\n".join(json.dumps(r) for r in result)+"\n")
 print("TOP3_AUDIT_RESULT",json.dumps({k:v for k,v in summary.items() if k!="examples"},sort_keys=True),flush=True)
if __name__=="__main__":main()
