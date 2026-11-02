"""Pinned Enca corpus: label-vetted and source-overlap-excluded, no tuning."""
import hashlib,json,codecs,statistics,time
from pathlib import Path
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_l3_language import LanguageCoherence
from l2_variant_matrix import valid

ALIASES={"utf8":"utf-8","cp866":"cp866","koi8r":"koi8-r","maccyr":"mac_cyrillic","iso88595":"iso8859-5","iso88592":"iso8859-2","iso88594":"iso8859-4","iso885913":"iso8859-13","iso885916":"iso8859-16","cp1251":"cp1251","cp1250":"cp1250","cp1257":"cp1257","ibm855":"cp855","ibm852":"cp852","ibm775":"cp775","macce":"mac_latin2","ucs2":"utf-16-be","ucs2le":"utf-16-le","ucs2be":"utf-16-be","ucs4":"utf-32-be","ucs4le":"utf-32-le","utf7":"utf-7"}
def main():
 rows=corpus(Path("training"));train=[(b,y,g) for b,y,g,k,sha in rows]
 seen={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
 prefixes={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
 det=fit(residual_subset(train)[0],"binary_high");language=LanguageCoherence.fit(train)
 stats={"found":0,"eligible_label":0,"overlap":0,"invalid_truth":0,"retained":0,"base_correct":0,"l3_correct":0,"fixed":0,"broken":0,"changed":0}
 cases=[];times=[]
 for path in sorted(Path("external/enca/test").iterdir()):
  if not path.is_file() or "." not in path.name:continue
  stats["found"]+=1;parts=path.name.split(".")
  if len(parts)!=2:continue
  enc=ALIASES.get(parts[-1])
  if not enc:continue
  try:codecs.lookup(enc)
  except LookupError:continue
  stats["eligible_label"]+=1
  b=path.read_bytes();digest=hashlib.sha256(b).hexdigest();prefix=hashlib.sha256(b[:4096]).hexdigest()
  if digest in seen or prefix in prefixes:stats["overlap"]+=1;continue
  truth=valid(b,enc)
  if truth is None:stats["invalid_truth"]+=1;continue
  start=time.process_time_ns();r=det.rank(b,final=len(b)<4096);mid=time.process_time_ns()
  q=language.rerank(r,b,final=len(b)<4096,max_l2_margin=.5,min_language_advantage=.15);end=time.process_time_ns()
  a=valid(b,r.encoding)==truth;c=valid(b,q.encoding)==truth
  stats["retained"]+=1;stats["base_correct"]+=int(a);stats["l3_correct"]+=int(c)
  stats["fixed"]+=int(not a and c);stats["broken"]+=int(a and not c);stats["changed"]+=int(q.changed)
  times.append(((mid-start)/1e6,(end-mid)/1e6))
  cases.append({"file":path.name,"label":enc,"base":r.encoding,"l3":q.encoding,"base_correct":a,"l3_correct":c,"l3_reason":q.reason})
 result={"stats":stats,"base_cpu_p50_ms":statistics.median(x[0] for x in times) if times else None,"l3_extra_cpu_p50_ms":statistics.median(x[1] for x in times) if times else None,"cases":cases,"warning":"Enca fixture labels and same-source sibling variants may be correlated; unique bytes do not imply independent source text. No tuning or promotion."}
 out=Path("research-results/enca-external");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(result,indent=2))
 print("ENCA_EXTERNAL_RESULT",json.dumps({k:v for k,v in result.items() if k!="cases"}),flush=True)
 print("ENCA_ERRORS",json.dumps([x for x in cases if not x["l3_correct"]][:15]),flush=True)
if __name__=="__main__":main()
