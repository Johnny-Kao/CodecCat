"""One-shot multi-variant L2 experiment. Same frozen folds, independent comparisons.
No deployment, no special-case codepage rules, no automatic promotion.
"""
import json,os,resource,time,statistics,codecs
from collections import defaultdict,Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier,RidgeClassifier
from sklearn.svm import LinearSVC
import numpy as np
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector

VARIANTS={
 "baseline":("residual",(2,3),45000,"sgd"),
 "full_train":("full",(2,3),45000,"sgd"),
 "unigram_bigram":("residual",(1,2),60000,"sgd"),
 "ngram_1_3":("residual",(1,3),90000,"sgd"),
 "ngram_2_4":("residual",(2,4),70000,"sgd"),
 "ridge":("residual",(2,3),45000,"ridge"),
 "linear_svc":("residual",(2,3),45000,"svc"),
 "balanced_12":("residual",(1,2),60000,"sgd_balanced"),
 "binary_12":("residual",(1,2),60000,"sgd_binary"),
 "wide_12":("residual",(1,2),90000,"sgd"),
 "strong_regularization":("residual",(2,3),45000,"sgd_strong"),
}
def valid(raw,enc):
 if not enc:return None
 try:return codecs.getincrementaldecoder(enc)("strict").decode(raw,final=len(raw)<4096).lstrip("\ufeff")
 except (UnicodeError,LookupError,ValueError,TypeError):return None
def run_variant(variant,fold,rows,external):
 mode,grams,features,kind=VARIANTS[variant]
 train=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
 if mode=="residual":train=residual_subset(train)[0]
 begin=time.process_time()
 if variant=="baseline":
  model=TrainableDetector.fit(train,feature_profile="stable23")
 else:
  labels=[y for b,y,g in train];strings=[b.decode("latin1") for b,y,g in train]
  vec=TfidfVectorizer(analyzer="char",ngram_range=grams,min_df=2,max_features=features,sublinear_tf=True,binary=(kind=="sgd_binary"))
  X=vec.fit_transform(strings)
  if kind=="ridge":clf=RidgeClassifier(alpha=1.0)
  elif kind=="svc":clf=LinearSVC(C=0.5,random_state=20261015)
  else:clf=SGDClassifier(loss="modified_huber",alpha=0.001 if kind=="sgd_strong" else 0.0001,random_state=20261015,max_iter=1500,class_weight="balanced" if kind=="sgd_balanced" else None)
  clf.fit(X,labels)
  model=TrainableDetector(vectorizer=vec,classifier=clf,fingerprint="experiment")
 train_cpu=time.process_time()-begin
 def evaluate(items):
  predictions=[];lat=[]
  for b,y,g,sha in items:
   start=time.process_time_ns()
   result=model.rank(b,final=len(b)<4096)
   lat.append(time.process_time_ns()-start)
   expected=valid(b,y);actual=valid(b,result.encoding)
   truth_rank=next((i+1 for i,x in enumerate(result.candidates) if valid(b,x)==expected),None)
   predictions.append({"sha":sha,"truth":y,"pred":result.encoding,"correct":expected==actual,"truth_rank":truth_rank})
  return predictions,lat
 held=[(b,y,g,sha) for b,y,g,k,sha in rows if k==fold]
 local,lat=evaluate(held)
 ext,elat=evaluate(external) if fold==0 else ([],[])
 return {"variant":variant,"fold":fold,"train_cpu_s":round(train_cpu,3),
  "local":local,"external":ext,"local_cpu_ms_p50":round(statistics.median(lat)/1e6,4),
  "local_cpu_ms_p95":round(sorted(lat)[int((len(lat)-1)*.95)]/1e6,4),
  "peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2)}
def main():
 import sys
 variant=sys.argv[1];out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
 rows=corpus(Path("training"))
 train_shas={sha for b,y,g,k,sha in rows}
 external=[]
 from mixed_random_benchmark import LABELS as labels
 for name,y in labels.items():
  b=(Path("external/charset_normalizer/data")/name).read_bytes()
  import hashlib
  sha=hashlib.sha256(b[:4096]).hexdigest()
  if sha not in train_shas and valid(b,y) is not None:external.append((b,y,name,sha))
 results=[run_variant(variant,i,rows,external) for i in range(3)]
 local=[x for r in results for x in r["local"]];ext=results[0]["external"]
 family=defaultdict(lambda:Counter())
 for row in local:
  family[row["truth"]]["n"]+=1;family[row["truth"]]["correct"]+=int(row["correct"])
 stats={"variant":variant,"local_n":len(local),"local_correct":sum(x["correct"] for x in local),
  "external_n":len(ext),"external_correct":sum(x["correct"] for x in ext),
  "recoverable_top3":sum((not x["correct"]) and x["truth_rank"] is not None for x in local),
  "truth_absent_top3":sum((not x["correct"]) and x["truth_rank"] is None for x in local),
  "train_cpu_s":round(sum(x["train_cpu_s"] for x in results),2),
  "cpu_p50_ms":round(statistics.median(x["local_cpu_ms_p50"] for x in results),4),
  "cpu_p95_ms":round(max(x["local_cpu_ms_p95"] for x in results),4),
  "peak_rss_mib":max(x["peak_rss_mib"] for x in results),
  "by_encoding":{k:dict(v) for k,v in sorted(family.items())}}
 (out/(variant+".json")).write_text(json.dumps({"summary":stats,"cases":local,"external":ext},indent=2))
 print("VARIANT_RESULT",json.dumps({k:v for k,v in stats.items() if k!="by_encoding"}),flush=True)
if __name__=="__main__":main()
