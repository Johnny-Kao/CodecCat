"""Separate semantic false-acceptance diagnostics and bounded repeated-call RSS.
No heuristic fitted, no production runtime changed.
"""
import hashlib,json,random,resource,time,gc,statistics
from pathlib import Path
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from input_robustness_gate import binary_model
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_decoding import decode

def negatives():
 rng=random.Random(20261102)
 items=[]
 signatures=[
   ("png",bytes.fromhex("89504e470d0a1a0a")),
   ("jpeg",bytes.fromhex("ffd8ffe000104a464946")),
   ("zip",b"PK\x03\x04"),
   ("gzip",bytes.fromhex("1f8b0800")),
   ("elf",b"\x7fELF\x02\x01\x01"),
   ("pdf",b"%PDF-1.7\n"),
   ("wasm",b"\x00asm\x01\x00\x00\x00"),
 ]
 for name,s in signatures:
  for n in (128,1024,4096,8192):
   items.append((name,s+rng.randbytes(n)))
 for n in (64,256,1024,4096,8192):
  for k in range(12):items.append(("uniform_random",rng.randbytes(n)))
 for n in (64,256,1024,4096,8192):
  items.append(("zero_padded",b"\x00"*n))
  items.append(("alternating_bytes",b"\x00\xff"*(n//2)))
 return items
def rss_kib():
 return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
def main():
 rows=corpus(Path("training"))
 train=residual_subset([(b,y,g) for b,y,g,k,sha in rows if k!=0])[0]
 positives=[(y,b) for b,y,g,k,sha in rows if k==0][:400]
 neg=negatives()
 result={"positive_n":len(positives),"negative_n":len(neg),"negative_categories":sorted(set(n for n,_ in neg)),"models":{},"limitations":"Synthetic binary negatives are diagnostic only; PDF etc may contain text. A detector must not necessarily reject all such containers. All memory values are process high-water marks including training."}
 for name,model in (("stable23",TrainableDetector.fit(train)),("binary_alpha_high",binary_model(train))):
  grouped={}
  for cat,b in neg:
   d=decode(model,b,bom_policy="strip")
   grouped.setdefault(cat,{"n":0,"accepted":0,"ambiguous":0})
   grouped[cat]["n"]+=1;grouped[cat]["accepted"]+=int(d.accepted);grouped[cat]["ambiguous"]+=int(d.ambiguous)
  positive_accepted=sum(decode(model,b,bom_policy="strip").accepted for _,b in positives)
  batch=[b for _,b in positives[:100]]+[b for _,b in neg[:50]]
  t=time.process_time();first=[(r.encoding,r.reason,r.ambiguous) for b in batch for r in [decode(model,b,bom_policy="strip")]]
  for _ in range(12):
   current=[(r.encoding,r.reason,r.ambiguous) for b in batch for r in [decode(model,b,bom_policy="strip")]]
   assert current==first,name
  result["models"][name]={"positive_accepted":positive_accepted,"negative_by_category":grouped,"repeat_calls":13*len(batch),"repeat_consistent":True,"repeat_cpu_s":round(time.process_time()-t,3),"process_peak_rss_mib":round(rss_kib()/1024,2)}
 out=Path("research-results/semantic-stability");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(result,indent=2))
 print("SEMANTIC_STABILITY_RESULT",json.dumps(result),flush=True)
if __name__=="__main__":main()
