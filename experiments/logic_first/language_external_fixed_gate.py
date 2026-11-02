"""Frozen external-source gate for cross-fold-selected Unicode reranker.
Never use external examples to tune parameters. Does not change runtime.
"""
import hashlib,json,statistics,time,resource
from pathlib import Path
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from mixed_random_benchmark import LABELS,decode as truth_decode
from codeccat.vnext_l3_language import LanguageCoherence

def main():
 rows=corpus(Path("training"))
 training=[(b,y,g) for b,y,g,k,_ in rows]
 seen={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
 seen_prefix={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
 det=fit(residual_subset(training)[0],"binary_high")
 language=LanguageCoherence.fit(training)
 results=[];excluded=[];base_times=[];extra_times=[]
 for name,y in sorted(LABELS.items()):
  raw=(Path("external/charset_normalizer/data")/name).read_bytes()
  sha=hashlib.sha256(raw).hexdigest();prefix=hashlib.sha256(raw[:4096]).hexdigest()
  if sha in seen or prefix in seen_prefix:
   excluded.append({"name":name,"reason":"exact or inference-prefix overlap"});continue
  truth=truth_decode(raw,y)
  if truth is None:
   excluded.append({"name":name,"reason":"invalid labeled decoding"});continue
  t=time.process_time_ns();ranked=det.rank(raw,final=len(raw)<4096); t1=time.process_time_ns()
  chosen=language.rerank(ranked,raw,final=len(raw)<4096,max_l2_margin=.5,min_language_advantage=.15);t2=time.process_time_ns()
  base_ok=truth_decode(raw,ranked.encoding)==truth
  new_ok=truth_decode(raw,chosen.encoding)==truth
  base_times.append((t1-t)/1e6);extra_times.append((t2-t1)/1e6)
  results.append({"name":name,"truth":y,"baseline":ranked.encoding,"reranked":chosen.encoding,"base_correct":base_ok,"new_correct":new_ok,"changed":chosen.changed,"reason":chosen.reason})
 result={"external_n":len(results),"excluded_n":len(excluded),"excluded":excluded,
  "baseline_correct":sum(x["base_correct"] for x in results),"rerank_correct":sum(x["new_correct"] for x in results),
  "fixed":sum(not x["base_correct"] and x["new_correct"] for x in results),
  "broken":sum(x["base_correct"] and not x["new_correct"] for x in results),
  "rerank_changed":sum(x["changed"] for x in results),
  "base_cpu_p50_ms":statistics.median(base_times) if base_times else None,
  "rerank_extra_cpu_p50_ms":statistics.median(extra_times) if extra_times else None,
  "peak_process_rss_mib_includes_training":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
  "cases":results,"warning":"Fixture filenames are labels, not independently audited text ground truth. Small external holdout cannot establish generalization. No parameter choices made on these results."}
 out=Path("research-results/language-external");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(result,indent=2))
 print("LANGUAGE_EXTERNAL_RESULT",json.dumps({k:v for k,v in result.items() if k not in ("cases","excluded")}),flush=True)
 print("LANGUAGE_EXTERNAL_CASES",json.dumps(results),flush=True)
if __name__=="__main__":main()
