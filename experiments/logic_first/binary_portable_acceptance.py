"""Validate binary model serialization, legacy compatibility and fresh-process roundtrip."""
import json,sys,hashlib,subprocess,tempfile
from pathlib import Path
from l2_variant_matrix import VARIANTS,run_variant
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
def fit(rows,mode):
 train=residual_subset([(b,y,g) for b,y,g,k,_ in rows if k!=0])[0]
 if mode=="legacy":return TrainableDetector.fit(train)
 vec=TfidfVectorizer(analyzer="char",ngram_range=(1,2),binary=True,min_df=2,max_features=60000,sublinear_tf=True)
 x=vec.fit_transform([b.decode("latin1") for b,y,g in train])
 clf=SGDClassifier(loss="modified_huber",alpha=.0001,random_state=20261015,max_iter=1500)
 clf.fit(x,[y for b,y,g in train])
 return TrainableDetector(vectorizer=vec,classifier=clf,fingerprint="binary-fixture")
def main():
 rows=corpus(Path("training"));test=[b for b,y,g,k,sha in rows if k==0]
 out=Path("research-results/binary-portable");out.mkdir(parents=True,exist_ok=True)
 for mode in ("binary","legacy"):
  model=fit(rows,mode);path=out/(mode+".zip")
  model.save_portable(path)
  loaded=TrainableDetector.load_portable(path)
  predictions=[model.rank(b,final=len(b)<4096).encoding for b in test]
  roundtrip=[loaded.rank(b,final=len(b)<4096).encoding for b in test]
  assert predictions==roundtrip,(mode,"in-process mismatch")
  import zipfile
  with zipfile.ZipFile(path) as z: meta=json.loads(z.read("manifest.json"))
  assert meta["params"].get("binary",False)==(mode=="binary")
  compare=subprocess.run([sys.executable,__file__,"--compare",str(path)],input=json.dumps([b.hex() for b in test]),text=True,capture_output=True,check=True)
  assert predictions==json.loads(compare.stdout),(mode,"fresh-process mismatch")
  print("PORTABLE_RESULT",json.dumps({"mode":mode,"n":len(test),"matching":len(test),"archive_bytes":path.stat().st_size,"schema":meta["schema"]}),flush=True)
if __name__=="__main__":
 if len(sys.argv)>1 and sys.argv[1]=="--compare":
  model=TrainableDetector.load_portable(sys.argv[2])
  b=[bytes.fromhex(v) for v in json.loads(sys.stdin.read())]
  print(json.dumps([model.rank(x,final=len(x)<4096).encoding for x in b]))
 else:main()
