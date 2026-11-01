"""Independent-source check: randomly ordered original charset-normalizer fixtures.

This is a small 18-document test, not a large random independent corpus.
"""
import json,random,sys,codecs,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
from charset_normalizer import from_bytes
import chardet,charset_normalizer

labels={"sample-arabic-1.txt":"cp1256","sample-french-1.txt":"cp1252","sample-arabic.txt":"utf_8","sample-russian-3.txt":"utf_8","sample-french.txt":"utf_8","sample-chinese.txt":"big5","sample-greek.txt":"cp1253","sample-greek-2.txt":"cp1253","sample-hebrew-2.txt":"cp1255","sample-hebrew-3.txt":"cp1255","sample-bulgarian.txt":"utf_8","sample-english.bom.txt":"utf_8","sample-spanish.txt":"utf_8","sample-korean.txt":"cp949","sample-turkish.txt":"cp1254","sample-russian-2.txt":"utf_8","sample-russian.txt":"mac_cyrillic","sample-polish.txt":"utf_8"}

def decode(raw,encoding):
 if not encoding:return None
 try:return codecs.getincrementaldecoder(encoding)("strict").decode(raw[:4096],final=len(raw)<4096).lstrip("\ufeff")
 except (UnicodeError,LookupError,ValueError,TypeError):return None

def main():
 train=corpus(Path("training"))
 residual,_=residual_subset([(b,y,g) for b,y,g,_,_ in train])
 model=TrainableDetector.fit(residual)
 paths=list(labels)
 random.Random(20261102).shuffle(paths)
 results=[]
 for name in paths:
  raw=(Path("external/charset_normalizer/data")/name).read_bytes()
  truth=labels[name]
  expected=decode(raw,truth)
  if expected is None:raise ValueError("invalid truth "+name)
  p=model.rank(raw,final=len(raw)<4096).encoding
  cp=chardet.detect(raw).get("encoding")
  best=from_bytes(raw).best();np=best.encoding if best else None
  results.append({"name":name,"raw_sha256":hashlib.sha256(raw).hexdigest(),"truth":truth,
   "codec_guess":p,"chardet_guess":cp,"normalizer_guess":np,
   "codec":decode(raw,p)==expected,"chardet":decode(raw,cp)==expected,"normalizer":decode(raw,np)==expected})
 report={"source":"jawah/charset_normalizer/data, pinned repository",
 "seed":20261102,"n":len(results),"correct":{k:sum(x[k] for x in results) for k in ("codec","chardet","normalizer")},
 "chardet_version":chardet.__version__,"normalizer_version":charset_normalizer.__version__,
 "limits":["18 independent-source fixture files, not a large random-sampled corpus",
 "charset-normalizer's own test examples favor it; prior overlap and semantic similarity with training corpus not exhaustively eliminated",
 "Fixed random order does not create random underlying sample."]}
 out=Path("research-results/charset-normalizer-origin");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(report,indent=2))
 (out/"cases.json").write_text(json.dumps(results,indent=2))
 print("CN_SOURCE_START");print(json.dumps(report,indent=2));print("CN_SOURCE_END")
if __name__=="__main__":main()
