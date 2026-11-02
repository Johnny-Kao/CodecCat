"""Fuzz-like deterministic adversarial input and steady-state contract gate."""
import json,random,hashlib,sys,subprocess,os,time
from pathlib import Path
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
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
def main():
 root=Path("research-results/input-robustness");root.mkdir(parents=True,exist_ok=True)
 rows=corpus(Path("training"))
 train=residual_subset([(b,y,g) for b,y,g,k,sha in rows if k!=0])[0]
 model=TrainableDetector.fit(train)
 path=root/"model.zip";model.save_portable(path)
 inp=cases();a=run(model,inp);b=run(model,inp);assert a==b
 report={"n":len(inp),"repeat_parity":True,"accepted":sum(r[1] is not None for r in a),"abstained":sum(r[1] is None for r in a),"max_len":max(map(len,inp))}
 (root/"report.json").write_text(json.dumps(report,indent=2))
 print("ROBUSTNESS_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
