"""Compare deterministic declared charset decoding with base over 3 folds and Enca."""
import codecs,hashlib,json,collections,time,statistics
from pathlib import Path
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from enca_external_gate import ALIASES
from l2_variant_matrix import valid
from codeccat.vnext_declarations import choose
def evaluate():
 rows=corpus(Path("training"))
 metrics={};reasons={}
 for fold in range(4):
  if fold<3:
   train=[(b,y,g) for b,y,g,k,sha in rows if k!=fold]
   test=[(b,y,g) for b,y,g,k,sha in rows if k==fold]
   source="development"
  else:
   train=[(b,y,g) for b,y,g,k,sha in rows]
   seen={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
   prefixes={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
   test=[]
   for p in sorted(Path("external/enca/test").iterdir()):
    if not p.is_file() or len(p.name.split("."))!=2:continue
    label=ALIASES.get(p.name.split(".")[-1])
    if not label:continue
    try:codecs.lookup(label)
    except LookupError:continue
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest() in seen or hashlib.sha256(raw[:4096]).hexdigest() in prefixes:continue
    if valid(raw,label) is None:continue
    test.append((raw,label,p.name))
   source="enca"
  model=fit(residual_subset(train)[0],"binary_high")
  c=metrics.setdefault(source,collections.Counter());times=[]
  for raw,y,g in test:
   ranked=model.rank(raw,final=len(raw)<4096)
   truth=valid(raw,y)
   base=valid(raw,ranked.encoding)==truth
   start=time.process_time_ns()
   q=choose(raw,ranked.encoding)
   times.append((time.process_time_ns()-start)/1e6)
   ok=valid(raw,q.encoding)==truth
   c["n"]+=1;c["base_correct"]+=int(base);c["chosen_correct"]+=int(ok)
   c["fixed"]+=int(not base and ok);c["broken"]+=int(base and not ok)
   c["changed"]+=int(q.encoding!=ranked.encoding)
   if base and not ok:
    reasons.setdefault(source,[]).append({"type":"regression","truth":y,"group":g,"baseline":ranked.encoding,"chosen":q.encoding,"reason":q.reason})
   if not base and ok:
    c["fixed_by_"+q.reason]+=1
   if q.reason.startswith("declared-"):
    c["declaration_found"]+=1
    if not ok:reasons.setdefault(source,[]).append({"truth":y,"baseline":ranked.encoding,"chosen":q.encoding,"reason":q.reason})
  c["cost_p50_us"]=round(statistics.median(times)*1000,3)
 # Synthetic sanity invariants, not accuracy claims
 fixtures={
  "xml_latin2":(b'<?xml version="1.0" encoding="iso-8859-2"?><x>\xa1</x>',"iso8859-2"),
  "html_cp1252":(b'<meta charset="windows-1252"><p>\x80</p>',"cp1252"),
  "css_utf8":(b'@charset "utf-8";\nspan{content:"\xc3\xa9"}',"utf-8"),
 }
 examples={}
 for name,(raw,y) in fixtures.items():
  result=choose(raw,"utf-8")
  examples[name]={"encoding":result.encoding,"correct":valid(raw,result.encoding)==valid(raw,y),"reason":result.reason}
 result={"sources":{k:dict(v) for k,v in metrics.items()},"failures":reasons,"synthetic_contract_examples":examples,
  "warning":"Declaration may be spoofed by untrusted document; only use within documented content semantics. Synthetic contract tests do not estimate real-world accuracy."}
 out=Path("research-results/declaration-evidence");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(result,indent=2))
 print("DECLARATION_EVIDENCE_RESULT",json.dumps({k:v for k,v in result.items() if k!="failures"}),flush=True)
if __name__=="__main__":evaluate()
