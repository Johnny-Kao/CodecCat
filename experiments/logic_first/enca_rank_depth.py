"""Find truth-equivalent candidate ranks beyond Top3 on Enca cases.
No model/policy change. Full legal-candidate ranking is diagnostic only.
"""
import codecs,hashlib,collections,json
from pathlib import Path
import numpy as np
from enca_external_gate import ALIASES
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from l2_variant_matrix import valid
def main():
 rows=corpus(Path("training"));training=[(b,y,g) for b,y,g,k,sha in rows]
 seen={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
 seen_p={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
 model=fit(residual_subset(training)[0],"binary_high")
 counts=collections.Counter();found=[];buckets=collections.Counter()
 for path in sorted(Path("external/enca/test").iterdir()):
  if not path.is_file() or len(path.name.split("."))!=2:continue
  label=ALIASES.get(path.name.split(".")[-1])
  if not label:continue
  try:codecs.lookup(label)
  except LookupError:continue
  raw=path.read_bytes()
  if hashlib.sha256(raw).hexdigest() in seen or hashlib.sha256(raw[:4096]).hexdigest() in seen_p:continue
  truth=valid(raw,label)
  if truth is None:continue
  r=model.rank(raw,final=len(raw)<4096)
  if valid(raw,r.encoding)==truth:continue
  x=model.vectorizer.transform([raw[:4096].decode("latin1")]).getrow(0)
  scores=np.asarray(model.classifier.coef_[:,x.indices] @ x.data+model.classifier.intercept_)
  names=list(model.classifier.classes_)
  order=np.argsort(scores)[::-1]
  legal_rank=0;rank=None
  for idx in order:
   candidate=str(names[int(idx)])
   if valid(raw,candidate) is None:continue
   legal_rank+=1
   if valid(raw,candidate)==truth:
    rank=legal_rank
    break
  counts["wrong"]+=1
  if rank is None:counts["absent_from_trained_classes"]+=1;bucket="absent"
  elif rank<=3:counts["top3"]+=1;bucket="top3"
  elif rank<=10:counts["rank4_to10"]+=1;bucket="4-10"
  else:counts["beyond10"]+=1;bucket="11+"
  buckets[(bucket,str(r.encoding))]+=1
  found.append({"file":path.name,"label":label,"pred":r.encoding,"equivalent_rank":rank,"trained":label in names,"byte_length":len(raw)})
 out=Path("research-results/enca-rank-depth");out.mkdir(parents=True,exist_ok=True)
 report={"counts":dict(counts),"by_prediction":[{"group":a,"prediction":b,"n":n} for (a,b),n in buckets.most_common()],"cases":found,"caveat":"Ranks use full bytes for truth-validity, whereas model candidate eligibility uses first 4096 prefix; for longer documents ranks can differ. This is an offline audit only."}
 (out/"report.json").write_text(json.dumps(report,indent=2))
 print("ENCA_RANK_DEPTH_RESULT",json.dumps(report["counts"]),flush=True)
if __name__=="__main__":main()
