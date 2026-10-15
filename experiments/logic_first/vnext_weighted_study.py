"""Paired development ablation: deterministic L1-easy examples receive lower
weight, residual examples higher weight, without dropping label coverage.
"""
import argparse,collections,json,sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from codeccat.vnext_strong_l2 import TrainableDetector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import stable_structural_guess,residual_subset

class WeightedDetector(TrainableDetector):
    @classmethod
    def fit_weighted(cls,rows,easy_weight):
        labels={y for _,y,_ in rows}
        vec=TfidfVectorizer(analyzer="char",ngram_range=(2,3),min_df=2,
                            max_features=45000,sublinear_tf=True)
        X=vec.fit_transform([b[:4096].decode("latin1") for b,_,_ in rows])
        y=[label for _,label,_ in rows]
        weights=np.array([easy_weight if stable_structural_guess(b,labels)==label else 1.0
                          for b,label,_ in rows],dtype=float)
        clf=SGDClassifier(loss="modified_huber",alpha=0.0001,
                          random_state=20261015,max_iter=1500)
        clf.fit(X,y,sample_weight=weights)
        return cls(vectorizer=vec,classifier=clf,fingerprint="weighted-dev")

def run(args):
    rows=corpus(Path(args.fixtures))
    all_rows=[]
    for fold in range(3):
        train=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
        held=[(b,y,g,sha) for b,y,g,k,sha in rows if k==fold]
        assert {g for _,_,g in train}.isdisjoint({g for _,_,g,_ in held})
        trimmed,_=residual_subset(train)
        detectors={"full":TrainableDetector.fit(train),
                   "capped_prune":TrainableDetector.fit(trimmed),
                   "weighted_0_35":WeightedDetector.fit_weighted(train,0.35),
                   "weighted_0_65":WeightedDetector.fit_weighted(train,0.65)}
        for b,truth,g,sha in held:
            values={name:det.rank(b,final=len(b)<4096).encoding for name,det in detectors.items()}
            all_rows.append({"fold":fold,"sha":sha,"truth":truth,**values})
        print("WEIGHTED_FOLD_DONE",fold,flush=True)
    n=len(all_rows)
    results={}
    for name in ("full","capped_prune","weighted_0_35","weighted_0_65"):
        hits=sum(x[name]==x["truth"] for x in all_rows)
        results[name]={"hits":hits,"accuracy":round(hits/n,6),
          "by_fold":[sum(x[name]==x["truth"] for x in all_rows if x["fold"]==k) for k in range(3)]}
    for name in ("weighted_0_35","weighted_0_65"):
        results[name]["helps_vs_prune"]=sum(x[name]==x["truth"] and x["capped_prune"]!=x["truth"] for x in all_rows)
        results[name]["harms_vs_prune"]=sum(x[name]!=x["truth"] and x["capped_prune"]==x["truth"] for x in all_rows)
    out={"n":n,"models":results,"note":"competitor-owned corpus, grouped 3fold, developmental experiment; no independent acceptance"}
    path=Path(args.output);path.mkdir(parents=True,exist_ok=True)
    (path/"report.json").write_text(json.dumps(out,indent=2))
    (path/"paired.jsonl").write_text("\n".join(json.dumps(x) for x in all_rows)+"\n")
    print("WEIGHTED_RESULT_BEGIN");print(json.dumps(out,indent=2));print("WEIGHTED_RESULT_END")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True)
    p.add_argument("--output",default="research-results/vnext-weighted-study");run(p.parse_args())
