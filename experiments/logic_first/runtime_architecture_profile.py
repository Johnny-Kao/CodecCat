"""One-shot inference cost attribution. Uses a frozen grouped training fold.
Reports timings and RSS independently for distinct inference stages.
"""
import json, time, statistics, resource, gc, os
from pathlib import Path
import numpy as np
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector

def meter(fn,inputs):
 times=[]
 for x in inputs:
  t=time.process_time_ns()
  fn(x)
  times.append((time.process_time_ns()-t)/1e6)
 return {"n":len(times),"cpu_p50_ms":round(statistics.median(times),5),
         "cpu_p95_ms":round(sorted(times)[int(.95*(len(times)-1))],5),
         "cpu_total_ms":round(sum(times),3)}
def main():
 rows=corpus(Path("training"))
 train=residual_subset([(b,y,g) for b,y,g,k,sha in rows if k!=0])[0]
 model=TrainableDetector.fit(train,feature_profile="stable23")
 test=[b for b,y,g,k,sha in rows if k==0]
 test=test[:min(500,len(test))]
 for b in test[:20]:model.rank(b,final=len(b)<4096)
 gc.collect()
 def vectorize(b):return model.vectorizer.transform([b[:4096].decode("latin1")])
 sparse=[vectorize(b) for b in test]
 def classify(row):
  row=row.getrow(0)
  margins=np.asarray(model.classifier.coef_[:,row.indices] @ row.data + model.classifier.intercept_)
  return np.argsort(margins)[::-1]
 def gate(b):return model.rank(b,final=len(b)<4096)
 from codeccat.vnext_gate import evaluate
 classes=tuple(str(x) for x in model.classifier.classes_)
 def all_decode(b):return evaluate(b[:4096],classes,final=len(b)<=4096)
 result={"n":len(test),"vectorize":meter(vectorize,test),
  "sparse_linear_score_sort":meter(classify,sparse),
  "full_rank_including_valid_decode":meter(gate,test),
  "all_class_structural_evidence":meter(all_decode,test),
  "peak_process_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
  "warning":"CPU per stage measured sequentially; full rank includes model gate and decoding. RSS includes sklearn training and imports, not inference-only."}
 out=Path("research-results/runtime-profile");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(result,indent=2))
 print("RUNTIME_PROFILE_RESULT",json.dumps(result),flush=True)
if __name__=="__main__":main()
