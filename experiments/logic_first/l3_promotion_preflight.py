"""Selective L3 promotion preflight: external fixtures, roundtrip, latency, regression."""
import argparse,codecs,hashlib,json,statistics,sys,tempfile,time,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_training_strategy_audit import strict_text
from external_accuracy_tournament import truth_label
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_l3_language import LanguageCoherence

def main():
 p=argparse.ArgumentParser();p.add_argument("--training",required=True);p.add_argument("--external",required=True);p.add_argument("--out",default="research-results/l3-promotion-preflight");a=p.parse_args()
 rows=corpus(Path(a.training));train=residual_subset([(b,y,g) for b,y,g,_,_ in rows])[0]
 model=TrainableDetector.fit(train); l3=LanguageCoherence.fit(train)
 with tempfile.TemporaryDirectory() as d:
  file=Path(d)/"l3.json";l3.save(file);reloaded=LanguageCoherence.load(file)
  assert reloaded.fingerprint==l3.fingerprint
  l3bytes=file.stat().st_size
  examples=[];base_ns=[];l3_ns=[]
  train_text={hashlib.sha256(strict_text(b,y).replace("\r\n","\n").replace("\r","\n").lstrip("\ufeff").encode()).hexdigest() for b,y,_,_,_ in rows if strict_text(b,y) is not None}
  for path in sorted((Path(a.external)/"test").rglob("*")):
   if not path.is_file():continue
   label=truth_label(path)
   if label is None:continue
   raw=path.read_bytes();expected=strict_text(raw,label)
   if expected is None:continue
   th=hashlib.sha256(expected.replace("\r\n","\n").replace("\r","\n").lstrip("\ufeff").encode()).hexdigest()
   if th in train_text:continue
   final=len(raw)<4096
   st=time.perf_counter_ns();rank=model.rank(raw,final=final);base_ns.append(time.perf_counter_ns()-st)
   st=time.perf_counter_ns();res=reloaded.rerank(rank,raw,final=final);l3_ns.append(time.perf_counter_ns()-st)
   check=l3.rerank(rank,raw,final=final)
   assert (check.encoding,check.called,check.changed,check.reason)==(res.encoding,res.called,res.changed,res.reason)
   before=strict_text(raw,rank.encoding);after=strict_text(raw,res.encoding)
   examples.append({"path":path.relative_to(a.external).as_posix(),"truth":label,
      "baseline":rank.encoding,"selective":res.encoding,"reason":res.reason,
      "baseline_correct":before==expected if before is not None else False,
      "selective_correct":after==expected if after is not None else False,"called":res.called,"changed":res.changed})
  assert examples
  n=len(examples);base=sum(x["baseline_correct"] for x in examples);new=sum(x["selective_correct"] for x in examples)
  help_=sum(x["selective_correct"] and not x["baseline_correct"] for x in examples)
  harm=sum(x["baseline_correct"] and not x["selective_correct"] for x in examples)
  report={"schema":"codeccat-l3-preflight-v1","status":"development-external-fixtures-NOT-blind",
   "n":n,"baseline_correct":base,"selective_correct":new,"help":help_,"harm":harm,
   "called":sum(x["called"] for x in examples),"changed":sum(x["changed"] for x in examples),
   "language_model_bytes":l3bytes,"language_roundtrip_cases":n,
   "timing_ns":{"baseline_p50":statistics.median(base_ns),"l3_extra_p50":statistics.median(l3_ns),
    "baseline_p95":sorted(base_ns)[int((n-1)*.95)],"l3_extra_p95":sorted(l3_ns)[int((n-1)*.95)]},
   "process_peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
   "limitations":["Third-party fixtures are not independent verified originals.",
    "RSS includes training and imports, not inference-only isolated RSS.",
    "Timing from 21 heterogeneous files is exploratory, not comparative throughput.",
    "No automatic promotion."]}
  out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
  (out/"summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  (out/"cases.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True) for x in examples)+"\n")
  print("L3_PREFLIGHT_BEGIN",flush=True);print(json.dumps(report,indent=2),flush=True);print("L3_PREFLIGHT_END",flush=True)
if __name__=="__main__":main()
