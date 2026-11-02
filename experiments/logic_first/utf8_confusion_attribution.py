"""Test whether Enca UTF-8 confusions are structural or ranking failures.
Fixed model, no learned parameter tuning or production changes.
"""
import json,hashlib,codecs,collections
from pathlib import Path
from enca_external_gate import ALIASES
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from l2_variant_matrix import valid

def main():
 rows=corpus(Path("training"));train=[(b,y,g) for b,y,g,k,sha in rows]
 model=fit(residual_subset(train)[0],"binary_high")
 seen={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
 prefixes={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
 counts=collections.Counter();cases=[]
 for path in sorted(Path("external/enca/test").iterdir()):
  if not path.is_file() or len(path.name.split("."))!=2:continue
  label=ALIASES.get(path.name.split(".")[-1])
  if not label:continue
  try:codecs.lookup(label)
  except LookupError:continue
  raw=path.read_bytes()
  if hashlib.sha256(raw).hexdigest() in seen or hashlib.sha256(raw[:4096]).hexdigest() in prefixes:continue
  truth=valid(raw,label)
  if truth is None:continue
  result=model.rank(raw,final=len(raw)<4096)
  if valid(raw,result.encoding)==truth:continue
  counts["wrong"]+=1
  prefix=raw[:4096]
  utf8_valid=valid(prefix,"utf-8") is not None
  predicted_utf8=(result.encoding or "").replace("_","-").startswith("utf-8")
  correct_available=label in [str(x) for x in model.classifier.classes_]
  candidates=[x for x in result.candidates if valid(raw,x)==truth]
  counts["utf8_pred"]+=int(predicted_utf8)
  counts["utf8_pred_invalid_bytes"]+=int(predicted_utf8 and not utf8_valid)
  counts["utf8_pred_valid_bytes"]+=int(predicted_utf8 and utf8_valid)
  counts["truth_label_trained"]+=int(correct_available)
  counts["truth_text_in_top3"]+=int(bool(candidates))
  cases.append({"file":path.name,"truth":label,"pred":result.encoding,"utf8_prefix_valid":utf8_valid,"truth_label_trained":correct_available,"truth_in_top3":bool(candidates),"bytes":len(raw)})
 out=Path("research-results/utf8-attribution");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps({"counts":dict(counts),"cases":cases},indent=2))
 print("UTF8_ATTRIBUTION_RESULT",json.dumps(dict(counts)),flush=True)
 print("UTF8_ATTRIBUTION_CASES",json.dumps(cases[:20]),flush=True)
if __name__=="__main__":main()
