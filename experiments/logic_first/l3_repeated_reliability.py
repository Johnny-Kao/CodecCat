"""Repeated controlled L3 audit: no tuning, multi-seed, paired cases and repeats."""
import argparse,hashlib,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_training_strategy_audit import strict_text
from external_accuracy_tournament import truth_label
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_l3_language import LanguageCoherence

def normalized(text):
 return hashlib.sha256(text.replace("\r\n","\n").replace("\r","\n").lstrip("\ufeff").encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--training",required=True);p.add_argument("--external",required=True);p.add_argument("--out",default="research-results/l3-repeated-audit");a=p.parse_args()
 rows=corpus(Path(a.training))
 known={normalized(strict_text(b,y)) for b,y,_,_,_ in rows if strict_text(b,y) is not None}
 external=[]
 for path in sorted((Path(a.external)/"test").rglob("*")):
  if not path.is_file():continue
  label=truth_label(path)
  if not label:continue
  raw=path.read_bytes();reference=strict_text(raw,label)
  if reference is None or normalized(reference) in known:continue
  external.append((path.relative_to(a.external).as_posix(),raw,label,reference))
 results=[]
 for arm in ("full","residual"):
  train=[(b,y,g) for b,y,g,_,_ in rows]
  if arm=="residual":train=residual_subset(train)[0]
  for replicate in range(3):
   model=TrainableDetector.fit(train)
   lang=LanguageCoherence.fit(train)
   cases=[];base_times=[];extra_times=[]
   for path,raw,truth,reference in external:
    final=len(raw)<4096
    start=time.perf_counter_ns();rank=model.rank(raw,final=final);base_times.append(time.perf_counter_ns()-start)
    start=time.perf_counter_ns();choice=lang.rerank(rank,raw,final=final);extra_times.append(time.perf_counter_ns()-start)
    before=strict_text(raw,rank.encoding)==reference
    after=strict_text(raw,choice.encoding)==reference
    cases.append({"path":path,"before":before,"after":after,"changed":choice.changed,
                  "prediction":rank.encoding,"after_prediction":choice.encoding})
   wrong=[x["path"] for x in cases if x["before"] and not x["after"]]
   results.append({"training":arm,"replicate":replicate,"n":len(cases),
                   "baseline_correct":sum(x["before"] for x in cases),
                   "l3_correct":sum(x["after"] for x in cases),
                   "helps":sum(x["after"] and not x["before"] for x in cases),
                   "harms":sum(x["before"] and not x["after"] for x in cases),
                   "changed":sum(x["changed"] for x in cases),
                   "harm_cases":wrong,"baseline_median_ns":statistics.median(base_times),
                   "l3_extra_median_ns":statistics.median(extra_times),
                   "cases":cases})
   print("L3_REPEAT",arm,replicate,len(external),wrong,flush=True)
 report={"schema":"codeccat-l3-repeated-v1","runs":results,
 "invariant_across_repeats":{arm:len({json.dumps([(c["path"],c["prediction"],c["after_prediction"]) for c in r["cases"]]) for r in results if r["training"]==arm})==1 for arm in ("full","residual")},
 "warnings":["Three repeated fits per strategy are deterministic repeats, not independent samples or different random seeds.",
 "Measurements are exploratory, use one CI runner, and include Python scheduling noise.",
 "No independently provenance-certified external source; no model promotion."]}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
 print("L3_REPEAT_REPORT_START");print(json.dumps({**report,"runs":[{k:v for k,v in r.items() if k!="cases"} for r in results]},indent=2));print("L3_REPEAT_REPORT_END")
if __name__=="__main__":main()
