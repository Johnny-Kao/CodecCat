"""Real opt-in layered runtime smoke: strict correctness, deterministic cases and negatives."""
import collections,json,time,statistics
from pathlib import Path
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from semantic_stability_audit import negatives
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence
from l2_variant_matrix import valid

def main():
 rows=corpus(Path("training"))
 totals=collections.Counter()
 elapsed=[]
 for fold in range(3):
  train=[(b,y,g) for b,y,g,k,sha in rows if k!=fold]
  model=fit(residual_subset(train)[0],"binary_high")
  language=LanguageCoherence.fit(train)
  for raw,label,g,k,sha in rows:
   if k!=fold:continue
   t=time.process_time_ns()
   result=detect(model,raw,language_model=language)
   elapsed.append((time.process_time_ns()-t)/1e6)
   expected=valid(raw,label)
   totals["positive"]+=1
   totals["accepted"]+=int(result.accepted)
   # Strip BOM when comparing against truth (same logical text contract).
   actual=result.text
   if expected is not None and expected.startswith("\ufeff"):expected=expected[1:]
   totals["correct"]+=int(actual==expected)
   if not result.accepted:totals["rejected"]+=1
   if result.declaration_used:totals["declarations"]+=1
   if result.language_changed:totals["language_changed"]+=1
 train=[(b,y,g) for b,y,g,k,sha in rows if k!=0]
 model=fit(residual_subset(train)[0],"binary_high")
 language=LanguageCoherence.fit(train)
 for name,raw in negatives():
  result=detect(model,raw,language_model=language)
  totals["synthetic_negatives"]+=1
  totals["synthetic_rejected"]+=int(not result.accepted)
 result={"counts":dict(totals),
        "total_pipeline_cpu_p50_ms":statistics.median(elapsed),
        "total_pipeline_cpu_p95_ms":sorted(elapsed)[int((len(elapsed)-1)*.95)],
        "warning":"Research pipeline. Synthetic binaries not representative. Development fold policy chosen on same corpus. No production integration."}
 out=Path("research-results/layered-runtime-smoke");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(result,indent=2))
 print("LAYERED_RUNTIME_SMOKE",json.dumps(result),flush=True)
if __name__=="__main__":main()
