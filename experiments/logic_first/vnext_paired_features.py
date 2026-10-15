"""Paired source-group CV: baseline vs candidate, no external holdout tuning.

Report source-family and individual label changes, score same bytes with same
folds, and reject candidate if it harms a major family despite aggregate gain.
"""
import argparse,collections,codecs,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_l2_feature_ablation import CONFIGS,fit,evaluate as original_evaluate

def fam(s):
    if s.startswith(("utf-","utf_")):return "unicode"
    if s.startswith(("iso8859","cp12","mac","koi8","hp-")):return "single_byte"
    if s.startswith(("gb","big5","shift","euc","iso2022","cp932","cp949")):return "east_asian"
    return "other"
def same_text(b,t,p):
    try:
        q=len(b)<4096
        return codecs.getincrementaldecoder(t)("strict").decode(b,final=q)==codecs.getincrementaldecoder(p)("strict").decode(b,final=q)
    except (LookupError,UnicodeError,TypeError):return False
def main(args):
    rows=corpus(Path(args.train))
    paired=[];folds=[]
    for k in range(3):
        tr=[(b,y,g) for b,y,g,fold,h in rows if fold!=k]
        te=[(b,y,g,sha) for b,y,g,fold,sha in rows if fold==k]
        assert {x[2] for x in tr}.isdisjoint({x[2] for x in te})
        models={name:fit(tr,CONFIGS[name]) for name in ("baseline","coverage_13")}
        summary=collections.Counter()
        raw_evaluation={(b,y,g) for b,y,g,sha in te}
        del raw_evaluation
        reference={name:original_evaluate(m,[(b,y,g) for b,y,g,sha in te]) for name,m in models.items()}
        for b,y,g,sha in te:
            pred={name:model.rank(b,final=len(b)<4096) for name,model in models.items()}
            a=same_text(b,y,pred["baseline"].encoding)
            z=same_text(b,y,pred["coverage_13"].encoding)
            r={"sha":sha,"fold":k,"group":g,"family":fam(y),"truth":y,
               "baseline":pred["baseline"].encoding,"candidate":pred["coverage_13"].encoding,
               "old_correct":a,"new_correct":z,
               "old_top3":y in pred["baseline"].candidates,
               "new_top3":y in pred["coverage_13"].candidates}
            paired.append(r)
            summary["n"]+=1
            summary["old_correct"]+=int(a);summary["new_correct"]+=int(z)
        summary["reference_old"]=reference["baseline"]["text"]
        summary["reference_new"]=reference["coverage_13"]["text"]
        assert summary["old_correct"]==summary["reference_old"], ("baseline metric divergence",k,summary)
        assert summary["new_correct"]==summary["reference_new"], ("candidate metric divergence",k,summary)
        folds.append({"fold":k,**dict(summary)})
        print("PAIRED_FOLD_DONE",k,flush=True)
    by_family={};by_label={}
    for group_name,key in ((by_family,"family"),(by_label,"truth")):
        for label in sorted({x[key] for x in paired}):
            q=[x for x in paired if x[key]==label]
            group_name[label]={
                "n":len(q),"before":sum(x["old_correct"] for x in q),
                "after":sum(x["new_correct"] for x in q),
                "gained":sum(x["new_correct"] and not x["old_correct"] for x in q),
                "lost":sum(x["old_correct"] and not x["new_correct"] for x in q),
                "top3_before":sum(x["old_top3"] for x in q),
                "top3_after":sum(x["new_top3"] for x in q)}
    result={"n":len(paired),"folds":folds,"family":by_family,
        "labels":by_label,
        "overall":{"before":sum(x["old_correct"] for x in paired),
                   "after":sum(x["new_correct"] for x in paired),
                   "gained":sum(x["new_correct"] and not x["old_correct"] for x in paired),
                   "lost":sum(x["old_correct"] and not x["new_correct"] for x in paired)},
        "note":"dev-only grouped CV. No independent external confirmation."}
    p=Path(args.output);p.mkdir(parents=True,exist_ok=True)
    (p/"paired-report.json").write_text(json.dumps(result,indent=2))
    (p/"paired-traces.jsonl").write_text("\n".join(map(json.dumps,paired))+"\n")
    print("PAIRED_RESULT_BEGIN");print(json.dumps(result,indent=2));print("PAIRED_RESULT_END")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--train",required=True)
    p.add_argument("--output",default="research-results/vnext-paired-features")
    main(p.parse_args())
