"""One-shot release candidate audit: paired accuracy and per-class regression.
No automatic promotion. Tests baseline and challenger in same process and folds.
"""
import json,hashlib,resource,statistics,time
from collections import Counter,defaultdict
from pathlib import Path
from l2_variant_matrix import run_variant
from vnext_actual_adapter import corpus
from mixed_random_benchmark import LABELS
def main():
 rows=corpus(Path("training"))
 seen={x[4] for x in rows};external=[]
 from l2_variant_matrix import valid
 for name,label in LABELS.items():
  b=(Path("external/charset_normalizer/data")/name).read_bytes()
  sha=hashlib.sha256(b[:4096]).hexdigest()
  if sha not in seen and valid(b,label) is not None:
   external.append((b,label,name,sha))
 variants={}
 for name in ("baseline","unigram_bigram"):
  runs=[run_variant(name,i,rows,external) for i in range(3)]
  vals=[x for r in runs for x in r["local"]]
  variants[name]={"folds":runs,"by_sha":{x["sha"]:x for x in vals},
    "external":{x["sha"]:x for x in runs[0]["external"]}}
  print("CANDIDATE_FIT_DONE",name,flush=True)
 a=variants["baseline"];b=variants["unigram_bigram"]
 assert a["by_sha"].keys()==b["by_sha"].keys()
 summary={"n":len(a["by_sha"]),"baseline_correct":0,"candidate_correct":0,
  "candidate_only":0,"baseline_only":0,"both_correct":0,"both_wrong":0}
 family=defaultdict(lambda:Counter());diffs=[]
 for sha,x in a["by_sha"].items():
  y=b["by_sha"][sha]
  assert x["truth"]==y["truth"]
  family[x["truth"]]["n"]+=1
  summary["baseline_correct"]+=int(x["correct"]);summary["candidate_correct"]+=int(y["correct"])
  family[x["truth"]]["baseline"]+=int(x["correct"]);family[x["truth"]]["candidate"]+=int(y["correct"])
  typ=("both_correct" if x["correct"] and y["correct"] else "both_wrong" if not x["correct"] and not y["correct"] else "baseline_only" if x["correct"] else "candidate_only")
  summary[typ]+=1
  if x["correct"]!=y["correct"]:diffs.append({"sha":sha,"truth":x["truth"],"baseline":x["pred"],"candidate":y["pred"],"result":typ})
 ext=[]
 for sha,x in a["external"].items():
  y=b["external"][sha]
  ext.append({"sha":sha,"truth":x["truth"],"baseline":x["correct"],"candidate":y["correct"],"baseline_pred":x["pred"],"candidate_pred":y["pred"]})
 regressions=[{"encoding":k,"n":v["n"],"baseline_correct":v["baseline"],"candidate_correct":v["candidate"],"delta":v["candidate"]-v["baseline"]} for k,v in family.items()]
 regressions.sort(key=lambda x:(x["delta"],x["encoding"]))
 summary["external_n"]=len(ext)
 summary["external_baseline_correct"]=sum(x["baseline"] for x in ext)
 summary["external_candidate_correct"]=sum(x["candidate"] for x in ext)
 summary["worst_regressions"]=regressions[:20]
 summary["best_gains"]=regressions[-20:][::-1]
 summary["n_regressed_families"]=sum(x["delta"]<0 for x in regressions)
 summary["n_improved_families"]=sum(x["delta"]>0 for x in regressions)
 summary["total_family_losses"]=sum(-x["delta"] for x in regressions if x["delta"]<0)
 summary["total_family_gains"]=sum(x["delta"] for x in regressions if x["delta"]>0)
 summary["speed_and_resource"]={k:{"train_cpu_s":sum(x["train_cpu_s"] for x in v["folds"]),
 "median_of_fold_p50_cpu_ms":statistics.median(x["local_cpu_ms_p50"] for x in v["folds"]),
 "max_fold_p95_cpu_ms":max(x["local_cpu_ms_p95"] for x in v["folds"]),
 "peak_process_rss_mib":max(x["peak_rss_mib"] for x in v["folds"])} for k,v in variants.items()}
 summary["promotion_decision"]="RESEARCH_ONLY: cross-source sample too small, model selected on development folds"
 out=Path("research-results/l2-candidate-final");out.mkdir(parents=True,exist_ok=True)
 (out/"report.json").write_text(json.dumps(summary,indent=2))
 (out/"paired_differences.json").write_text(json.dumps(diffs,indent=2))
 (out/"per_encoding.json").write_text(json.dumps(regressions,indent=2))
 (out/"external.json").write_text(json.dumps(ext,indent=2))
 print("CANDIDATE_AUDIT",json.dumps(summary),flush=True)
if __name__=="__main__":main()
