"""Real opt-in layered runtime smoke: strict correctness, deterministic cases and negatives."""
import collections,json,time,statistics,codecs,hashlib
from pathlib import Path
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from semantic_stability_audit import negatives
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence
from l2_variant_matrix import valid
from layered_architecture_gate import outside

def main():
 rows=corpus(Path("training"))
 totals=collections.Counter()
 mismatches=[]
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
   # Compare historical prefix-decoding score with a complete-document strict contract.
   try:
    strict_expected=codecs.getincrementaldecoder(label)("strict").decode(raw,final=True).lstrip("\ufeff")
   except (UnicodeError,LookupError):
    strict_expected=None
   totals["strict_truth_valid"]+=int(strict_expected is not None)
   if strict_expected is not None:totals["strict_complete_correct"]+=int(actual==strict_expected)
   prior=model.rank(raw,final=len(raw)<4096)
   current=model.rank(raw,final=True)
   totals["rank_finality_changed"]+=int(prior.encoding!=current.encoding)
   if raw and len(raw)==4096:totals["length_4096"]+=1
   selected_valid=valid(raw,result.encoding) if result.encoding else None
   if selected_valid is not None and selected_valid.startswith("\ufeff"):selected_valid=selected_valid[1:]
   if selected_valid==expected and actual!=expected:
    totals["selected_codec_decoding_contract_mismatch"]+=1
    if len(mismatches)<18:
     mismatches.append({"group":g,"truth":label,"selected":result.encoding,"reason":result.reason,
      "bom":raw[:8].hex(),"raw_len":len(raw),"expected_len":len(expected) if expected is not None else None,
      "actual_len":len(actual) if actual is not None else None,
      "expected_first":repr(expected[:10]) if expected is not None else None,
      "actual_first":repr(actual[:10]) if actual is not None else None})
   if not result.accepted and expected is not None:
    totals["rejected_text"]+=1
   if not result.accepted:totals["rejected"]+=1
   if result.declaration_used:totals["declarations"]+=1
   if result.language_changed:totals["language_changed"]+=1
 # Strict complete-decoding comparison on three external source repositories.
 external={}
 full_train=[(b,y,g) for b,y,g,k,sha in rows]
 full_model=fit(residual_subset(full_train)[0],"binary_high")
 full_language=LanguageCoherence.fit(full_train)
 known={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
 prefixes={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
 for source,name,raw,label in outside(rows):
  if hashlib.sha256(raw).hexdigest() in known or hashlib.sha256(raw[:4096]).hexdigest() in prefixes:continue
  try:expected_full=codecs.getincrementaldecoder(label)("strict").decode(raw,final=True).lstrip("\ufeff")
  except (UnicodeError,LookupError):continue
  d=detect(full_model,raw,language_model=full_language)
  baseline=full_model.rank(raw,final=True)
  try:base_text=codecs.getincrementaldecoder(baseline.encoding)("strict").decode(raw,final=True).lstrip("\ufeff")
  except (UnicodeError,LookupError,TypeError):base_text=None
  c=external.setdefault(source,collections.Counter())
  c["n"]+=1;c["base_correct"]+=int(base_text==expected_full);c["combined_correct"]+=int(d.text==expected_full)
  c["accepted"]+=int(d.accepted)
 # Existing test source includes an actual binary fixture from Enca.
 real=Path("external/enca/test/binary")
 if real.is_file():
  sample=real.read_bytes()
  check=detect(full_model,sample,language_model=full_language)
  totals["actual_enca_binary_n"]+=1
  totals["actual_enca_binary_rejected"]+=int(not check.accepted)
 train=[(b,y,g) for b,y,g,k,sha in rows if k!=0]
 model=fit(residual_subset(train)[0],"binary_high")
 language=LanguageCoherence.fit(train)
 for name,raw in negatives():
  result=detect(model,raw,language_model=language)
  totals["synthetic_negatives"]+=1
  totals["synthetic_rejected"]+=int(not result.accepted)
 result={"counts":dict(totals),"external":{key:dict(val) for key,val in external.items()},
        "total_pipeline_cpu_p50_ms":statistics.median(elapsed),
        "total_pipeline_cpu_p95_ms":sorted(elapsed)[int((len(elapsed)-1)*.95)],
        "warning":"Research pipeline. Synthetic binaries not representative. Development fold policy chosen on same corpus. No production integration."}
 out=Path("research-results/layered-runtime-smoke");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(result,indent=2))
 print("LAYERED_RUNTIME_SMOKE",json.dumps(result),flush=True)
 print("LAYERED_RUNTIME_DECODE_MISMATCH",json.dumps(mismatches),flush=True)
if __name__=="__main__":main()
