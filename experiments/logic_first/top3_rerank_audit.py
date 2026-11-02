"""Offline source-disjoint held-out Top3 rerank matrix, no weight changes."""
import json,math,collections
from pathlib import Path
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from l2_variant_matrix import valid

def quality(raw,enc):
 s=valid(raw,enc)
 if s is None:return -100.
 if not s:return 0.
 n=len(s)
 bad=sum((ord(c)<32 and c not in "\t\r\n") or 127<=ord(c)<160 for c in s)
 rare=sum(0xE000<=ord(c)<=0xF8FF or 0xD800<=ord(c)<=0xDFFF for c in s)
 return -(bad+rare*2)/n
def main():
 rows=corpus(Path("training"))
 variants={k:collections.Counter() for k in (0.,0.1,0.5,1.,2.,4.)}
 changes={k:[] for k in variants}
 for fold in range(3):
  train=residual_subset([(b,y,g) for b,y,g,k,_ in rows if k!=fold])[0]
  model=fit(train,"binary_high")
  for b,y,g,k,sha in rows:
   if k!=fold:continue
   result=model.rank(b,final=len(b)<4096)
   expected=valid(b,y)
   base=valid(b,result.encoding)==expected
   for weight in variants:
    choice=result.encoding
    if result.candidates:
     scored=[(score+weight*quality(b,label),-i,label) for i,(label,score) in enumerate(zip(result.candidates,result.scores))]
     choice=max(scored)[2]
    correct=valid(b,choice)==expected
    z=variants[weight]
    z["n"]+=1;z["correct"]+=int(correct)
    z["fixed"]+=int(correct and not base)
    z["broken"]+=int(base and not correct)
    z["changed"]+=int(choice!=result.encoding)
 report={"metric":"decoded text equality, grouped 3-fold development fixtures","variants":{str(k):dict(v) for k,v in variants.items()},"caution":"Weights evaluated on same development folds; selected best may overfit. Not a promotion gate."}
 out=Path("research-results/top3-rerank");out.mkdir(parents=True,exist_ok=True)
 (out/"summary.json").write_text(json.dumps(report,indent=2))
 print("RERANK_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
