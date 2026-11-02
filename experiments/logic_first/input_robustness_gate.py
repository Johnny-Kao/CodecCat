"""Fuzz-like deterministic adversarial input and steady-state contract gate."""
import json,random,hashlib,sys,subprocess,os,time
from pathlib import Path
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from codeccat.vnext_decoding import decode
def cases():
 rng=random.Random(20261102)
 v=[b"",bytes(range(256)),b"\x00"*65536,b"A"*131072,b"\xff"*131072,
    b"\xef\xbb\xbf",b"\xff\xfe",b"\xfe\xff",b"\x00\x00\xfe\xff",
    b"\xff\xfe\x00\x00",b"\x1b",b"\xc0\xaf",b"\xf0\x9f\x92",b"\xff\xff",
    b"\x00"*4095,b"\x00"*4096,b"\x00"*4097]
 for n in (1,2,3,7,16,31,32,63,64,255,256,4095,4096,4097,10000):
  for _ in range(12):
   v.append(rng.randbytes(n))
 return v
def run(model,inp):
 out=[]
 for data in inp:
  item=decode(model,data,bom_policy="strip")
  assert item.text is None or isinstance(item.text,str)
  if item.text is not None:
   assert "\ufffd" not in item.text or data.find(b"\xef\xbf\xbd")!=-1 or data.find(b"\xff\xfd")!=-1, "possible replacement"
  if item.encoding is not None: assert item.text is not None
  out.append((item.encoding,item.text,item.reason,item.ambiguous))
 return out
def binary_model(train):
 strings=[b.decode("latin1") for b,y,g in train]
 vec=TfidfVectorizer(analyzer="char",ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,binary=True)
 x=vec.fit_transform(strings)
 clf=SGDClassifier(loss="modified_huber",alpha=0.0003,random_state=20261015,max_iter=1500)
 clf.fit(x,[y for b,y,g in train])
 return TrainableDetector(vectorizer=vec,classifier=clf,fingerprint="robustness-binary")
def main():
 root=Path("research-results/input-robustness");root.mkdir(parents=True,exist_ok=True)
 rows=corpus(Path("training"))
 train=residual_subset([(b,y,g) for b,y,g,k,sha in rows if k!=0])[0]
 inp=cases()
 report={"n":len(inp),"max_len":max(map(len,inp)),"models":{}}
 for label,model in (("stable23",TrainableDetector.fit(train)),("binary_alpha_high",binary_model(train))):
  path=root/(label+".zip");model.save_portable(path)
  restored=TrainableDetector.load_portable(path)
  a=run(model,inp);b=run(model,inp);c=run(restored,inp)
  assert a==b==c,(label,"prediction drift across load or repeat")
  report["models"][label]={"repeat_and_reload_parity":True,"accepted":sum(r[1] is not None for r in a),"abstained":sum(r[1] is None for r in a),"model_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
 report["semantic_note"]="Syntactically decodable random bytes are not necessarily meaningful text; this gate tests crash resistance and reproducibility only."
 (root/"report.json").write_text(json.dumps(report,indent=2))
 print("ROBUSTNESS_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
