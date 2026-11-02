"""External holdout: Stable23 vs binary 1-2 gram, no training on external records."""
import hashlib,json,sys,codecs
from pathlib import Path
from l2_variant_matrix import valid
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
def model_fit(rows,binary):
 training=residual_subset([(b,y,g) for b,y,g,k,sha in rows])[0]
 if not binary:return TrainableDetector.fit(training)
 vec=TfidfVectorizer(analyzer="char",ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,binary=True)
 X=vec.fit_transform([b.decode("latin1") for b,y,g in training])
 clf=SGDClassifier(loss="modified_huber",alpha=.0001,random_state=20261015,max_iter=1500)
 clf.fit(X,[y for b,y,g in training])
 return TrainableDetector(vectorizer=vec,classifier=clf,fingerprint="research")
def canonical(x):
 try:return codecs.lookup(x).name
 except LookupError:return None
def main():
 rows=corpus(Path("training")); models={n:model_fit(rows,k) for n,k in (("stable23",False),("binary12",True))}
 trainingbytes={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
 trainingtexts=set()
 for b,y,g,k,sha in rows:
  t=valid(b,y)
  if t is not None:trainingtexts.add(hashlib.sha256(t.encode("utf-8")).hexdigest())
 cases=[];skips=[]
 for p in sorted(Path("external/uchardet/test").rglob("*")):
  if not p.is_file() or p.suffix not in (".txt",".be",".le"):continue
  name=p.name.replace(".be","-be").replace(".le","-le") if p.suffix in (".be",".le") else p.stem
  truth=canonical(name)
  if not truth:continue
  raw=p.read_bytes()[:4096]
  expected=valid(raw,truth)
  if expected is None:continue
  sha=hashlib.sha256(raw).hexdigest();txtsha=hashlib.sha256(expected.encode()).hexdigest()
  if sha in trainingbytes or txtsha in trainingtexts:
   skips.append(p.relative_to("external/uchardet").as_posix());continue
  result={"path":p.relative_to("external/uchardet").as_posix(),"sha":sha,"truth":truth}
  for name,m in models.items():
   prediction=m.rank(raw,final=len(raw)<4096).encoding
   result[name]={"correct":valid(raw,prediction)==expected,"pred":prediction}
  cases.append(result)
 assert cases,"no external cases"
 counts={k:sum(x[k]["correct"] for x in cases) for k in models}
 changed=[x for x in cases if x["stable23"]["correct"] != x["binary12"]["correct"]]
 summary={"n":len(cases),"counts":counts,"binary_only":sum(x["binary12"]["correct"] and not x["stable23"]["correct"] for x in changed),"stable_only":sum(x["stable23"]["correct"] and not x["binary12"]["correct"] for x in changed),"excluded_overlap":len(skips),"source":"BYVoid/uchardet test fixtures, labels from filenames, not a blind independent natural-document test"}
 out=Path("research-results/external-binary-check");out.mkdir(parents=True,exist_ok=True)
 (out/"summary.json").write_text(json.dumps(summary,indent=2))
 (out/"cases.json").write_text(json.dumps(cases,indent=2))
 print("EXTERNAL_BINARY_RESULT",json.dumps(summary),flush=True)
if __name__=="__main__":main()
