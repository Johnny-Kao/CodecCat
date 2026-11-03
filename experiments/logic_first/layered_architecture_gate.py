"""Frozen layered decision order: binary preflight -> document declaration -> language L3 -> L2.

Research-only. Main classifier, thresholds and evaluated source commits immutable.
"""
import codecs,hashlib,json,collections,statistics,time
from pathlib import Path
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from l2_variant_matrix import valid
from enca_external_gate import ALIASES
from mixed_random_benchmark import LABELS
from codeccat.vnext_l3_language import LanguageCoherence
from codeccat.vnext_declarations import choose
from codeccat.vnext_binary_preflight import inspect_text

def outside(rows):
 seen={hashlib.sha256(raw).hexdigest() for raw,y,g,k,sha in rows}
 prefix={hashlib.sha256(raw[:4096]).hexdigest() for raw,y,g,k,sha in rows}
 for path in sorted(Path("external/enca/test").iterdir()):
  if not path.is_file() or len(path.name.split("."))!=2:continue
  label=ALIASES.get(path.name.split(".")[-1])
  if not label:continue
  try:codecs.lookup(label)
  except LookupError:continue
  yield "enca",path.name,path.read_bytes(),label
 for path in sorted(Path("external/uchardet/test").rglob("*.txt")):
  label=path.stem.replace("-","_")
  try:codecs.lookup(label)
  except LookupError:continue
  yield "uchardet",str(path.relative_to(Path("external/uchardet/test"))),path.read_bytes(),label
 for name,label in LABELS.items():
  p=Path("external/charset_normalizer/data")/name
  yield "charset-normalizer",name,p.read_bytes(),label

def summarize(data):
 return {k:{**dict(v),"cpu_p50_ms":statistics.median(data["latency"][k]) if data["latency"][k] else None}
         for k,v in data["counts"].items()}

def main():
 rows=corpus(Path("training"))
 data={"counts":{},"latency":{}}
 exceptions=[]
 def evaluate(source,name,raw,label,model,language):
  truth=valid(raw,label)
  if truth is None:return
  ranked=model.rank(raw,final=len(raw)<4096)
  base=valid(raw,ranked.encoding)==truth
  start=time.process_time_ns()
  declaration=choose(raw,ranked.encoding)
  l3=language.rerank(ranked,raw,final=len(raw)<4096,max_l2_margin=.5,min_language_advantage=.15)
  decl_used=declaration.reason.startswith("declared-")
  # Explicit content declaration wins over speculative language evidence.
  final_encoding=declaration.encoding if decl_used else l3.encoding
  first_policy=declaration.encoding
  end=time.process_time_ns()
  decl_ok=valid(raw,first_policy)==truth
  l3_ok=valid(raw,l3.encoding)==truth
  combined_ok=valid(raw,final_encoding)==truth
  c=data["counts"].setdefault(source,collections.Counter())
  data["latency"].setdefault(source,[]).append((end-start)/1e6)
  c["n"]+=1;c["baseline"]+=int(base)
  for key,ok in (("declaration",decl_ok),("language",l3_ok),("combined",combined_ok)):
   c[key]+=int(ok)
   c[key+"_fix"]+=int(not base and ok)
   c[key+"_break"]+=int(base and not ok)
  c["declaration_count"]+=int(decl_used)
  c["combined_change"]+=int(final_encoding!=ranked.encoding)
  if (base and not combined_ok) or (not base and combined_ok):
   exceptions.append({"source":source,"name":name,"truth":label,"base":ranked.encoding,
     "decl":declaration.encoding,"l3":l3.encoding,"final":final_encoding,
     "decl_reason":declaration.reason,"l3_reason":l3.reason,
     "fixed":not base and combined_ok,"broken":base and not combined_ok})
 for fold in range(3):
  train=[(b,y,g) for b,y,g,k,sha in rows if k!=fold]
  model=fit(residual_subset(train)[0],"binary_high")
  l3=LanguageCoherence.fit(train)
  for raw,y,g,k,sha in rows:
   if k==fold:evaluate("development",sha,raw,y,model,l3)
 train=[(b,y,g) for b,y,g,k,sha in rows]
 model=fit(residual_subset(train)[0],"binary_high")
 l3=LanguageCoherence.fit(train)
 seen={hashlib.sha256(raw).hexdigest() for raw,y,g,k,sha in rows}
 prefixes={hashlib.sha256(raw[:4096]).hexdigest() for raw,y,g,k,sha in rows}
 for source,name,raw,label in outside(rows):
  if hashlib.sha256(raw).hexdigest() in seen or hashlib.sha256(raw[:4096]).hexdigest() in prefixes:continue
  evaluate(source,name,raw,label,model,l3)
 results={key:{**dict(v),"extra_cpu_p50_ms":statistics.median(data["latency"][key])} for key,v in data["counts"].items()}
 negatives=collections.Counter()
 from semantic_stability_audit import negatives as make_negatives
 for category,raw in make_negatives():
  check=inspect_text(raw)
  negatives["n"]+=1;negatives["rejected"]+=int(not check.text_possible)
 report={"sources":results,"binary_synthetic":dict(negatives),"changed_cases":exceptions,
 "notes":"Neither thresholds nor datasets were tuned after selection; no production integration. The same-source external fixture collections may contain correlated documents. Declarations can be forged: trust only document formats."}
 out=Path("research-results/layered-decision")
 out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(report,indent=2))
 print("LAYERED_DECISION_RESULT",json.dumps({k:v for k,v in report.items() if k!="changed_cases"}),flush=True)
 print("LAYERED_DECISION_REGRESSIONS",json.dumps([x for x in exceptions if x["broken"]]),flush=True)
if __name__=="__main__":main()
