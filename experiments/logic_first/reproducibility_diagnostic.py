"""Reproducibility and dominant misclassification diagnostics. Never modifies runtime."""
import json,os,subprocess,sys,hashlib,collections
from pathlib import Path
def worker():
 from mixed_random_benchmark import build,decode
 from vnext_residual_study import residual_subset
 from codeccat.vnext_strong_l2 import TrainableDetector
 rows,selected=build();targets=[r for r in selected if r["source"]=="chardet"]
 result=[]
 for fold in range(3):
  training=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
  model=TrainableDetector.fit(residual_subset(training)[0],feature_profile="stable23")
  for r in targets:
   if r["fold"]!=fold:continue
   rank=model.rank(r["raw"],final=len(r["raw"])<4096)
   result.append({"sha":r["sha"],"truth":r["label"],"pred":rank.encoding,
      "top3":rank.candidates,"correct":decode(r["raw"],rank.encoding)==decode(r["raw"],r["label"])})
 print(json.dumps(sorted(result,key=lambda x:x["sha"])))
def main():
 out=Path("research-results/mixed-random-benchmark");out.mkdir(parents=True,exist_ok=True)
 trials=[]
 for seed in ("0","1","17"):
  env=dict(os.environ,PYTHONHASHSEED=seed)
  p=subprocess.run([sys.executable,__file__,"--worker"],env=env,capture_output=True,text=True,check=True)
  rows=json.loads(p.stdout);trials.append(rows)
  print("REPRO_RUN",seed,"n",len(rows),"correct",sum(x["correct"] for x in rows),"fingerprint",hashlib.sha256(p.stdout.encode()).hexdigest(),flush=True)
 base={r["sha"]:r for r in trials[0]}
 changes=[]
 for i in range(1,len(trials)):
  for r in trials[i]:
   if r["pred"]!=base[r["sha"]]["pred"]:changes.append({"seed":i,"sha":r["sha"],"a":base[r["sha"]],"b":r})
 original=json.loads((out/"chardet.json").read_text())["cases"]
 competitor={x["sha"]:x for x in original}
 mismatch=[];examples=collections.Counter()
 for r in trials[0]:
  ch=competitor[r["sha"]]
  if ch["correct"] and not r["correct"]:
   examples[(r["truth"],r["pred"])]+=1
   mismatch.append(r)
 report={"seeds":["0","1","17"],"counts":[sum(x["correct"] for x in rows) for rows in trials],
 "changed_predictions":changes[:100],"changed_count":len(changes),
 "top_confusions":[{"truth":a,"predicted":b,"n":n} for (a,b),n in examples.most_common(40)],
 "chardet_only":len(mismatch),"exact_fingerprints":[hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest() for t in trials]}
 (out/"reproducibility.json").write_text(json.dumps(report,indent=2))
 print("REPRO_DIAGNOSTIC",json.dumps({"counts":report["counts"],"changed_count":report["changed_count"],"top_confusions":report["top_confusions"][:20]}),flush=True)
if __name__=="__main__":
 if len(sys.argv)>1:worker()
 else:main()
