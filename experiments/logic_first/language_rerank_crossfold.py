"""Cross-fold language-coherence reranking without changing the base detector.
All language statistics fit on fold training data; no held-out labels in decisions.
"""
import json,collections
from pathlib import Path
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from l2_variant_matrix import valid
from codeccat.vnext_l3_language import LanguageCoherence
def main():
 rows=corpus(Path("training"))
 thresholds=[(.2,.15),(.5,.15),(1.,.15),(.5,.4),(1.,.4),(2.,.4)]
 totals={str(p):collections.Counter() for p in thresholds}
 for fold in range(3):
  training=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
  det=fit(residual_subset(training)[0],"binary_high")
  language=LanguageCoherence.fit(training)
  for b,y,g,k,sha in rows:
   if k!=fold:continue
   r=det.rank(b,final=len(b)<4096)
   expected=valid(b,y);original=valid(b,r.encoding)==expected
   for p in thresholds:
    q=language.rerank(r,b,final=len(b)<4096,max_l2_margin=p[0],min_language_advantage=p[1])
    correct=valid(b,q.encoding)==expected
    d=totals[str(p)]
    d["n"]+=1;d["base_correct"]+=int(original);d["correct"]+=int(correct);d["changed"]+=int(q.changed);d["fixed"]+=int(not original and correct);d["broken"]+=int(original and not correct)
 out=Path("research-results/language-rerank");out.mkdir(parents=True,exist_ok=True)
 result={"variants":{k:dict(v) for k,v in totals.items()},"scope":"development corpus grouped heldout fold; language data fit only on fold training rows; no source-independent claim"}
 (out/"report.json").write_text(json.dumps(result,indent=2))
 print("LANGUAGE_RERANK_RESULT",json.dumps(result),flush=True)
if __name__=="__main__":main()
