"""Fixed-weight family layer paired gates across development & external sources."""
import collections,hashlib,codecs,json,statistics,time
from pathlib import Path
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from l2_variant_matrix import valid
from enca_external_gate import ALIASES
from codeccat.vnext_l3_language import LanguageCoherence
from codeccat.vnext_family_proposal import propose

POLICIES={"conservative":{"language_advantage":1.0,"min_script_delta":0.0},
          "script_strong":{"language_advantage":0.6,"min_script_delta":0.2}}
def experiment():
 rows=corpus(Path("training"));summaries={}
 cases_per_source={"development":[], "enca":[]}
 for fold in range(3):
  training=[(b,y,g) for b,y,g,k,sha in rows if k!=fold]
  model=fit(residual_subset(training)[0],"binary_high")
  lang=LanguageCoherence.fit(training)
  for raw,label,group,k,sha in rows:
   if k!=fold:continue
   ranked=model.rank(raw,final=len(raw)<4096)
   target=valid(raw,label)
   base=valid(raw,ranked.encoding)==target
   alt={}
   for variant,kwargs in POLICIES.items():
    t=time.process_time_ns()
    q=propose(model,lang,raw,ranked,max_candidates=32,**kwargs)
    elapsed=(time.process_time_ns()-t)/1e6
    alt[variant]={"ok":valid(raw,q.encoding)==target,"changed":q.changed,"ms":elapsed,"examined":q.examined}
   cases_per_source["development"].append((base,alt))
 # external: training on all chardet rows, with strict hash overlap removal
 training=[(b,y,g) for b,y,g,k,sha in rows]
 model=fit(residual_subset(training)[0],"binary_high");lang=LanguageCoherence.fit(training)
 exact={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
 prefix={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
 for path in sorted(Path("external/enca/test").iterdir()):
  if not path.is_file() or len(path.name.split("."))!=2:continue
  label=ALIASES.get(path.name.split(".")[-1])
  if not label:continue
  try:codecs.lookup(label)
  except LookupError:continue
  raw=path.read_bytes()
  if hashlib.sha256(raw).hexdigest() in exact or hashlib.sha256(raw[:4096]).hexdigest() in prefix:continue
  target=valid(raw,label)
  if target is None:continue
  ranked=model.rank(raw,final=len(raw)<4096)
  base=valid(raw,ranked.encoding)==target
  alt={}
  for variant,kwargs in POLICIES.items():
   t=time.process_time_ns()
   q=propose(model,lang,raw,ranked,max_candidates=32,**kwargs)
   alt[variant]={"ok":valid(raw,q.encoding)==target,"changed":q.changed,"ms":(time.process_time_ns()-t)/1e6,"examined":q.examined}
  cases_per_source["enca"].append((base,alt))
 for source,cases in cases_per_source.items():
  variants={}
  for variant in POLICIES:
   costs=[a[variant]["ms"] for _,a in cases]
   variants[variant]={"correct":sum(a[variant]["ok"] for _,a in cases),
        "fixed":sum(not baseline and a[variant]["ok"] for baseline,a in cases),
        "broken":sum(baseline and not a[variant]["ok"] for baseline,a in cases),
        "changed":sum(a[variant]["changed"] for _,a in cases),
        "extra_cpu_p50_ms":statistics.median(costs) if costs else None,
        "extra_cpu_p95_ms":sorted(costs)[int((len(costs)-1)*.95)] if costs else None}
  summaries[source]={"n":len(cases),"baseline_correct":sum(a for a,_ in cases),"policies":variants}
 result={"summary":summaries,"notes":"Both thresholds chosen a priori; Enca not used for policy selection. Development folds source-group disjoint. Costs include 32-codec scan, sklearn vector transform and L3 scoring, but exclude original L2 rank. No release changes."}
 out=Path("research-results/family-proposal");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(result,indent=2))
 print("FAMILY_PROPOSAL_RESULT",json.dumps(result),flush=True)
if __name__=="__main__":experiment()
