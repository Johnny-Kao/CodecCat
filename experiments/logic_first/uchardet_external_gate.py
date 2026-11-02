"""Immutable source-separated uchardet fixture gate; never tune chosen thresholds.
Labels are fixture filenames and must be independently assessed before promotion.
"""
import codecs,hashlib,json,statistics,time
from pathlib import Path
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from architecture_attribution import fit
from codeccat.vnext_l3_language import LanguageCoherence
from l2_variant_matrix import valid
def main():
 root=Path("external/uchardet/test")
 training_rows=corpus(Path("training"))
 training=[(b,y,g) for b,y,g,k,sha in training_rows]
 seen_full={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in training_rows}
 seen_prefix={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in training_rows}
 model=fit(residual_subset(training)[0],"binary_high")
 language=LanguageCoherence.fit(training)
 counts={"found":0,"invalid_label":0,"undecodable_truth":0,"overlap":0,"retained":0,"baseline_correct":0,"rerank_correct":0,"fixed":0,"broken":0,"changed":0}
 cases=[];overlap=[];cost=[]
 for path in sorted(root.rglob("*.txt")):
  counts["found"]+=1
  enc=path.stem.replace("-","_")
  try:codecs.lookup(enc)
  except LookupError:
   counts["invalid_label"]+=1;continue
  raw=path.read_bytes()
  target=valid(raw,enc)
  if target is None:
   counts["undecodable_truth"]+=1;continue
  digest=hashlib.sha256(raw).hexdigest();prefix=hashlib.sha256(raw[:4096]).hexdigest()
  if digest in seen_full or prefix in seen_prefix:
   counts["overlap"]+=1;overlap.append(str(path));continue
  before=time.process_time_ns()
  ranked=model.rank(raw,final=len(raw)<4096)
  after=time.process_time_ns()
  chosen=language.rerank(ranked,raw,final=len(raw)<4096,max_l2_margin=0.5,min_language_advantage=0.15)
  end=time.process_time_ns()
  base=valid(raw,ranked.encoding)==target;new=valid(raw,chosen.encoding)==target
  counts["retained"]+=1;counts["baseline_correct"]+=int(base);counts["rerank_correct"]+=int(new)
  counts["fixed"]+=int(not base and new);counts["broken"]+=int(base and not new);counts["changed"]+=int(chosen.changed)
  cost.append({"base_ms":(after-before)/1e6,"extra_ms":(end-after)/1e6})
  cases.append({"file":str(path.relative_to(root)),"expected":enc,"pred":ranked.encoding,"reranked":chosen.encoding,"base_correct":base,"rerank_correct":new,"l3_reason":chosen.reason})
 report={"counts":counts,"overlap_paths":overlap,"cost_p50_ms":{k:statistics.median(x[k] for x in cost) if cost else None for k in ("base_ms","extra_ms")},"cases":cases,"warning":"Uchardet fixture filenames are proxy truth; duplicates excluded by full/prefix SHA only, not near-duplicates or source-text correspondence. This is distinct repository but not a verified statistically independent population."}
 out=Path("research-results/uchardet-independent");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(report,indent=2))
 print("UCH_EXTERNAL_RESULT",json.dumps({k:v for k,v in report.items() if k not in ("cases","overlap_paths")}),flush=True)
 print("UCH_FAILURES",json.dumps([x for x in cases if not x["rerank_correct"]][:25]),flush=True)
if __name__=="__main__":main()
