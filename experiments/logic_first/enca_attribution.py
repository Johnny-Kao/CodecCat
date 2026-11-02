"""Enca error attribution by original decoded text and Top3 recoverability.
No parameter adjustment or shipping code changes.
"""
import hashlib,json,collections,codecs
from pathlib import Path
from enca_external_gate import ALIASES
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from l2_variant_matrix import valid
def main():
 rows=corpus(Path("training"));tr=[(b,y,g) for b,y,g,k,sha in rows]
 seen={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
 prefixes={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
 model=fit(residual_subset(tr)[0],"binary_high")
 groups=collections.defaultdict(list);out=[];pairs=collections.Counter()
 counts=collections.Counter()
 for path in sorted(Path("external/enca/test").iterdir()):
  if not path.is_file() or len(path.name.split("."))!=2:continue
  label=ALIASES.get(path.name.split(".")[-1])
  if not label:continue
  try:codecs.lookup(label)
  except LookupError:continue
  b=path.read_bytes()
  if hashlib.sha256(b).hexdigest() in seen or hashlib.sha256(b[:4096]).hexdigest() in prefixes:continue
  truth=valid(b,label)
  if truth is None:continue
  ranked=model.rank(b,final=len(b)<4096)
  good=valid(b,ranked.encoding)==truth
  positions=[i+1 for i,x in enumerate(ranked.candidates) if valid(b,x)==truth]
  group=hashlib.sha256(truth.encode("utf-8","surrogatepass")).hexdigest()
  groups[group].append(path.name)
  if not good:
   counts["wrong"]+=1
   counts["recoverable_top3"]+=int(bool(positions))
   counts["missing_top3"]+=int(not positions)
   pairs[(label,str(ranked.encoding))]+=1
  out.append({"file":path.name,"label":label,"pred":ranked.encoding,"correct":good,
              "group":group,"top3_truth_rank":positions[0] if positions else None,
              "length":len(b),"top3":list(ranked.candidates)})
 bygroup=collections.defaultdict(list)
 for x in out:bygroup[x["group"]].append(x)
 stats={"n":len(out),"correct":sum(x["correct"] for x in out),
  "unique_decoded_texts":len(bygroup),"groups_with_multiple_encoding_variants":sum(len(v)>1 for v in bygroup.values()),
  "cases_in_multi_variant_groups":sum(len(v) for v in bygroup.values() if len(v)>1),
  "wrong":counts["wrong"],"wrong_recoverable_top3":counts["recoverable_top3"],"wrong_missing_top3":counts["missing_top3"],
  "group_macro_accuracy":sum(sum(x["correct"] for x in v)/len(v) for v in bygroup.values())/len(bygroup),
  "confusions":[{"truth":a,"pred":b,"n":n} for (a,b),n in pairs.most_common(25)]}
 result={"summary":stats,"cases":out,"same_text_groups":[{"files":[x["file"] for x in v],"correct":sum(x["correct"] for x in v),"n":len(v)} for v in bygroup.values() if len(v)>1],"limitations":"Byte-hash de-dup only vs training data. Group by canonical decoded Unicode text within Enca; not a fully independent unseen population. Truth derives from filename aliases."}
 path=Path("research-results/enca-attribution");path.mkdir(parents=True,exist_ok=True)
 (path/"report.json").write_text(json.dumps(result,indent=2))
 print("ENCA_ATTRIBUTION_RESULT",json.dumps(stats),flush=True)
if __name__=="__main__":main()
