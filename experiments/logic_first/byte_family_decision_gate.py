"""Frozen score-gap selection gate on cheap 256-bin family proposals.
The original detector is unchanged; no truth-dependent inference logic.
"""
import codecs,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from enca_external_gate import ALIASES
from l2_variant_matrix import valid
from codeccat.vnext_byte_families import ByteFamilies,histogram,is_valid

POLICIES={
 "gap_0.05":0.05,
 "gap_0.10":0.10,
 "gap_0.20":0.20,
 "gap_0.35":0.35,
}
def choose(raw,ranked,families,*,min_gap):
 base=ranked.encoding
 if base is None or ranked.bom_hint or len(ranked.candidates)==1:
  return base,False
 names=families.names
 if base not in names:return base,False
 v=histogram(raw);v[:128]*=.25;v/=max(float(np.linalg.norm(v)),1e-12)
 similarity=families.centroids@v
 original=float(similarity[names.index(base)])
 order=np.argsort(similarity)[::-1]
 for ix in order[:16]:
  label=names[int(ix)]
  if label==base:break
  if float(similarity[int(ix)])-original < min_gap:break
  if is_valid(raw,label):
   return label,True
 return base,False

def source_run(rows,source):
 if source=="development":
  tests=[(fold,[(b,y,g) for b,y,g,k,sha in rows if k!=fold],[(b,y,g,sha) for b,y,g,k,sha in rows if k==fold]) for fold in range(3)]
 else:
  training=[(b,y,g) for b,y,g,k,sha in rows]
  known={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
  known_p={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
  test=[]
  for p in sorted(Path("external/enca/test").iterdir()):
   if not p.is_file() or len(p.name.split("."))!=2:continue
   label=ALIASES.get(p.name.split(".")[-1])
   if label is None:continue
   try:codecs.lookup(label)
   except LookupError:continue
   b=p.read_bytes()
   if hashlib.sha256(b).hexdigest() in known or hashlib.sha256(b[:4096]).hexdigest() in known_p:continue
   if valid(b,label) is None:continue
   test.append((b,label,p.name,p.name))
  tests=[(3,training,test)]
 results={k:{"n":0,"base_correct":0,"correct":0,"fixed":0,"broken":0,"changed":0} for k in POLICIES}
 overhead=[]
 for fold,train,test in tests:
  model=fit(residual_subset(train)[0],"binary_high")
  families=ByteFamilies.fit(train)
  for raw,label,group,sha in test:
   truth=valid(raw,label);ranked=model.rank(raw,final=len(raw)<4096)
   base=valid(raw,ranked.encoding)==truth
   for k,gap in POLICIES.items():
    t=time.process_time_ns()
    enc,changed=choose(raw,ranked,families,min_gap=gap)
    d=(time.process_time_ns()-t)/1e6
    if k=="gap_0.10":overhead.append(d)
    ok=valid(raw,enc)==truth
    x=results[k]
    x["n"]+=1;x["base_correct"]+=int(base);x["correct"]+=int(ok)
    x["fixed"]+=int(not base and ok);x["broken"]+=int(base and not ok);x["changed"]+=int(changed)
 results["cost_gap_0.10"]={"p50_ms":statistics.median(overhead) if overhead else None,"p95_ms":sorted(overhead)[int((len(overhead)-1)*.95)] if overhead else None}
 return results

def main():
 rows=corpus(Path("training"))
 report={"development":source_run(rows,"development"),"enca":source_run(rows,"enca"),
         "warning":"Thresholds are predetermined; Enca is source-correlated, filename-labeled stress data; outcomes are not promotion-ready. This is a distinct experimental decision policy."}
 out=Path("research-results/byte-family-decision");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(report,indent=2))
 print("BYTE_FAMILY_DECISION_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
