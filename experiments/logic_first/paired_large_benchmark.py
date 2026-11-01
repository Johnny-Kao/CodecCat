"""Paired 3-fold group-held-out benchmark against chardet; seeded order, no leakage."""
import codecs,hashlib,json,random,statistics,time
from collections import defaultdict
from pathlib import Path
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
import chardet

def decode(b,encoding):
 if not encoding:return None
 try:return codecs.getincrementaldecoder(encoding)("strict").decode(b,final=len(b)<4096).lstrip("\ufeff")
 except (LookupError,UnicodeError,TypeError,ValueError):return None

def main():
 rows=corpus(Path("training"))
 rng=random.Random(20261101)
 # grouped 3-fold prevents same source group crossing train/test; no test-data holdout independence claim
 cases=[]
 for fold in range(3):
  training=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
  hold=[(b,y,g,s) for b,y,g,k,s in rows if k==fold]
  assert {g for _,_,g in training}.isdisjoint({g for _,_,g,_ in hold})
  residual,_=residual_subset(training)
  model=TrainableDetector.fit(residual,feature_profile="stable23")
  rng.shuffle(hold)
  for b,y,g,s in hold:
   expected=decode(b,y)
   if expected is None:continue
   t=time.perf_counter_ns();pred=model.rank(b,final=len(b)<4096).encoding;a=time.perf_counter_ns()-t
   t=time.perf_counter_ns();guess=chardet.detect(b)["encoding"];t2=time.perf_counter_ns()-t
   ca=decode(b,pred)==expected;cb=decode(b,guess)==expected
   cases.append({"fold":fold,"group":g,"sha256":s,"truth":y,"bytes":len(b),
     "codeccat":pred,"chardet":guess,"cc_correct":ca,"ch_correct":cb,
     "cc_ns":a,"ch_ns":t2})
  print("FOLD_COMPLETED",fold,len(hold),flush=True)
 def metrics(items):
  n=len(items)
  return {"n":n,"codeccat_correct":sum(x["cc_correct"] for x in items),
   "chardet_correct":sum(x["ch_correct"] for x in items),
   "both_correct":sum(x["cc_correct"] and x["ch_correct"] for x in items),
   "codec_only":sum(x["cc_correct"] and not x["ch_correct"] for x in items),
   "chardet_only":sum(x["ch_correct"] and not x["cc_correct"] for x in items)}
 families=defaultdict(list)
 for row in cases:families[row["truth"]].append(row)
 report={"schema":"codeccat-group-heldout-paired-v1","seed":20261101,
  "training_corpus":"chardet/test-data @ b0c0d206913ba118ffea2f7d306883e7a4a76db7",
  "model":"residual stable23 retrained separately on other two folds",
  "chardet_version":chardet.__version__,"overall":metrics(cases),
  "by_fold":{str(i):metrics([r for r in cases if r["fold"]==i]) for i in range(3)},
  "by_encoding":{k:metrics(v) for k,v in sorted(families.items())},
  "median_latency_ns":{"codeccat":statistics.median(x["cc_ns"] for x in cases),
   "chardet":statistics.median(x["ch_ns"] for x in cases)},
  "important_limits":["Not a blind independent-source test; chardet owns the fixture repository.",
  "Randomized evaluation order, not new random independent documents.",
  "Prefix corpus may end inside multibyte sequence: nonfinal decoding is used for 4096-byte chunks.",
  "Comparative latency includes Python runtime but omits install/import and model load.",
  "Some different codec labels produce the same Unicode text."]}
 out=Path("research-results/paired-large-benchmark");out.mkdir(parents=True,exist_ok=True)
 (out/"summary.json").write_text(json.dumps(report,indent=2,sort_keys=True))
 (out/"cases.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True) for x in cases)+"\n")
 print("PAIRED_REPORT_START");print(json.dumps({k:v for k,v in report.items() if k!="by_encoding"},indent=2));print("PAIRED_REPORT_END")
if __name__=="__main__":main()
