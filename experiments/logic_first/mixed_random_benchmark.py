"""Reproducible mixed-source stratified sample, exact leakage checks and process-level profiling."""
import argparse,codecs,hashlib,json,random,resource,statistics,sys,time
from collections import defaultdict,Counter
from pathlib import Path
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
LABELS={"sample-arabic-1.txt":"cp1256","sample-french-1.txt":"cp1252","sample-arabic.txt":"utf_8","sample-russian-3.txt":"utf_8","sample-french.txt":"utf_8","sample-chinese.txt":"big5","sample-greek.txt":"cp1253","sample-greek-2.txt":"cp1253","sample-hebrew-2.txt":"cp1255","sample-hebrew-3.txt":"cp1255","sample-bulgarian.txt":"utf_8","sample-english.bom.txt":"utf_8","sample-spanish.txt":"utf_8","sample-korean.txt":"cp949","sample-turkish.txt":"cp1254","sample-russian-2.txt":"utf_8","sample-russian.txt":"mac_cyrillic","sample-polish.txt":"utf_8"}
SEED=20261102
def decode(b,label):
 if not label:return None
 try:return codecs.getincrementaldecoder(label)("strict").decode(b,final=len(b)<4096).lstrip("\ufeff")
 except (LookupError,UnicodeError,ValueError,TypeError):return None
def build():
 rows=corpus(Path("training")); pool=[];seen=set()
 for raw,label,group,fold,digest in rows:
  if decode(raw,label) is None:continue
  if digest in seen:continue
  seen.add(digest);pool.append({"raw":raw,"label":label,"source":"chardet","group":group,"fold":fold,"sha":digest})
 for name,label in LABELS.items():
  raw=(Path("external/charset_normalizer/data")/name).read_bytes()
  if decode(raw,label) is None:continue
  digest=hashlib.sha256(raw).hexdigest()
  if digest in seen:continue
  seen.add(digest);pool.append({"raw":raw,"label":label,"source":"charset-normalizer","group":name,"fold":-1,"sha":digest})
 # Stratify chardet samples across encoding classes and folds; select all 18 other-source documents.
 rng=random.Random(SEED); buckets=defaultdict(list)
 for r in pool:buckets[(r["source"],r["label"],r["fold"])].append(r)
 selected=[]
 for key,b in sorted(buckets.items()):
  rng.shuffle(b)
  selected+=b[:min(len(b),max(3,round(len(b)*.35)))] if key[0]=="chardet" else b
 rng.shuffle(selected)
 return rows,selected
def evaluate(engine,rows,selected):
 import chardet,charset_normalizer
 from charset_normalizer import from_bytes
 from codeccat.vnext_strong_l2 import TrainableDetector
 models={}
 if engine=="codeccat":
  for fold in range(3):
   training=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
   train=residual_subset(training)[0]
   models[fold]=TrainableDetector.fit(train,feature_profile="stable23")
 # Chardet sourced cases are held out at source-group level from the corresponding CodecCat fold.
 # Other source cases use model trained on all chardet corpus, excluding *exact duplicates*.
 if engine=="codeccat":
  models[-1]=TrainableDetector.fit(residual_subset([(b,y,g) for b,y,g,k,_ in rows])[0],feature_profile="stable23")
 out=[];lat=[]
 start=time.process_time_ns()
 for r in selected:
  b=r["raw"];t=time.process_time_ns()
  if engine=="codeccat":pred=models[r["fold"]].rank(b,final=len(b)<4096).encoding
  elif engine=="chardet":pred=chardet.detect(b)["encoding"]
  else:
   best=from_bytes(b).best();pred=best.encoding if best else None
  delta=time.process_time_ns()-t;lat.append(delta)
  out.append({"sha":r["sha"],"source":r["source"],"truth":r["label"],"bytes":len(b),
              "prediction":pred,"correct":decode(b,pred)==decode(b,r["label"]),"cpu_ns":delta})
 elapsed=time.process_time_ns()-start
 return {"engine":engine,"n":len(out),"correct":sum(x["correct"] for x in out),
  "cpu_inference_total_ms":round(elapsed/1e6,3),"cpu_inference_p50_ms":round(statistics.median(lat)/1e6,4),
  "cpu_inference_p95_ms":round(sorted(lat)[int((len(lat)-1)*.95)]/1e6,4),
  "process_peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
  "by_source":{k:{"n":sum(x["source"]==k for x in out),"correct":sum(x["correct"] for x in out if x["source"]==k)} for k in ("chardet","charset-normalizer")},
  "cases":out}
def main():
 import os
 rows,selected=build()
 root=Path("research-results/mixed-random-benchmark");root.mkdir(parents=True,exist_ok=True)
 manifest=[{"sha":r["sha"],"source":r["source"],"truth":r["label"],"fold":r["fold"],"group":r["group"]} for r in selected]
 (root/"manifest.json").write_text(json.dumps({"seed":SEED,"selected":manifest},indent=2))
 for engine in ("codeccat","chardet","charset-normalizer"):
  # Separate OS processes for isolated RSS, imports, model training.
  import subprocess
  result=subprocess.run([sys.executable,__file__,"--worker",engine],capture_output=True,text=True,check=True)
  (root/(engine+".json")).write_text(result.stdout)
  print("MIXED_RESULT",result.stdout.strip(),flush=True)
if __name__=="__main__":
 if len(sys.argv)>1 and sys.argv[1]=="--worker":
  data=evaluate(sys.argv[2],*build());print(json.dumps(data))
 else:main()
