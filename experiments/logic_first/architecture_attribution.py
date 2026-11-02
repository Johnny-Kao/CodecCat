"""Architecture attribution: fixed weights, compare normal vs strict structure-first decisions.
Diagnostic only: no model parameters, no public API changes.
"""
import codecs,json,time
from collections import Counter,defaultdict
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
from l2_variant_matrix import valid
def fit(train,kind):
 if kind=="stable23":return TrainableDetector.fit(train)
 vec=TfidfVectorizer(analyzer="char",ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,binary=True)
 x=vec.fit_transform([b[:4096].decode("latin1") for b,y,g in train])
 clf=SGDClassifier(loss="modified_huber",alpha=.0003,random_state=20261015,max_iter=1500)
 clf.fit(x,[y for b,y,g in train])
 return TrainableDetector(vectorizer=vec,classifier=clf,fingerprint="architecture-only")
def main():
 rows=corpus(Path("training"));out={}
 for kind in ("stable23","binary_high"):
  c=Counter();errors=defaultdict(Counter);examples=[]
  for fold in range(3):
   train=residual_subset([(b,y,g) for b,y,g,k,_ in rows if k!=fold])[0]
   model=fit(train,kind)
   for b,y,g,k,sha in rows:
    if k!=fold:continue
    raw=b[:4096];expected=valid(raw,y)
    a=model.rank(raw,final=len(raw)<4096)
    s=model.rank(raw,final=len(raw)<4096,structural_first=True)
    v=valid(raw,a.encoding)==expected
    sv=valid(raw,s.encoding)==expected
    c["n"]+=1;c["normal_correct"]+=int(v);c["structural_correct"]+=int(sv)
    c["structure_changed"]+=int(a.encoding!=s.encoding)
    c["structure_fixed"]+=int(not v and sv)
    c["structure_broke"]+=int(v and not sv)
    c["correct_in_top3"]+=int(not v and any(valid(raw,z)==expected for z in a.candidates))
    c["outside_top3"]+=int(not v and not any(valid(raw,z)==expected for z in a.candidates))
    c["wrong_no_candidate"]+=int(not v and a.encoding is None)
    if not v:
     errors[y]["n"]+=1
     errors[y]["structural_fix"]+=int(sv)
     if len(examples)<12:examples.append({"sha":sha,"truth":y,"pred":a.encoding,"structural_pred":s.encoding,"fixed":sv})
  out[kind]={"counts":dict(c),"largest_error_families":[{"codec":k,**dict(v)} for k,v in sorted(errors.items(),key=lambda kv:-kv[1]["n"])[:18]],"examples":examples}
 target=Path("research-results/architecture-attribution");target.mkdir(parents=True,exist_ok=True)
 (target/"report.json").write_text(json.dumps(out,indent=2))
 print("ARCHITECTURE_ATTRIBUTION",json.dumps({k:v["counts"] for k,v in out.items()}),flush=True)
if __name__=="__main__":main()
