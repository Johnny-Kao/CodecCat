"""Independent feature-family weights, tested only on group-disjoint dev folds.

Uses exactly the same TF-IDF vocabulary, fitting and SGD class labels as the
baseline 1–3gram; differs solely in numeric weighting of 1/2/3gram columns.
No external holdout can be used to select weights.
"""
import argparse,collections,codecs,json,sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from codeccat.vnext_strong_l2 import TrainableDetector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus

WEIGHTS={
 "baseline":(1.,1.,1.),
 "downweight_unigram":(.5,1.,1.),
 "upweight_unigram":(1.5,1.,1.),
 "upweight_trigram":(1.,1.,1.5),
}
def fit(rows,weights):
    raw=[b[:4096].decode("latin1") for b,y,g in rows]
    labels=[y for b,y,g in rows]
    vec=TfidfVectorizer(analyzer="char",ngram_range=(1,3),min_df=1,max_features=90000,sublinear_tf=True)
    mat=vec.fit_transform(raw)
    factors=np.fromiter((weights[len(gram)-1] for gram,idx in sorted(vec.vocabulary_.items(),key=lambda z:z[1])),dtype=np.float64)
    weighted=mat.multiply(factors).tocsr()
    from sklearn.preprocessing import normalize
    weighted=normalize(weighted)
    clf=SGDClassifier(loss="modified_huber",alpha=0.0001,random_state=20261015,max_iter=1500)
    clf.fit(weighted,labels)
    # Apply exact feature weights again during inference, then normalize.
    return vec,clf,factors
def predict(model,b):
    vec,clf,factors=model
    from sklearn.preprocessing import normalize
    row=vec.transform([b[:4096].decode("latin1")]).multiply(factors).tocsr()
    row=normalize(row)
    score=clf.decision_function(row)[0]
    order=np.argsort(np.atleast_1d(score))[::-1]
    if len(clf.classes_)==2:
        order=np.argsort(np.array([-float(score),float(score)]))[::-1]
    for idx in order:
        label=str(clf.classes_[int(idx)])
        try:
            codecs.getincrementaldecoder(label)("strict").decode(b[:4096],final=len(b)<4096)
        except (UnicodeError,LookupError):continue
        return label
    return None
def equivalent(b,y,p):
    try:return codecs.getincrementaldecoder(y)("strict").decode(b,final=len(b)<4096)==codecs.getincrementaldecoder(p)("strict").decode(b,final=len(b)<4096)
    except (LookupError,UnicodeError,TypeError):return False
def main(args):
    rows=corpus(Path(args.fixtures))
    report={"n":len(rows),"configs":{},"warning":"dev-fold-only exploration; not deployable until runtime portable feature-weight schema exists"}
    for name,weights in WEIGHTS.items():
        folds=[]
        for k in range(3):
            train=[(b,y,g) for b,y,g,f,h in rows if f!=k]
            test=[(b,y,g) for b,y,g,f,h in rows if f==k]
            assert {g for b,y,g in train}.isdisjoint({g for b,y,g in test})
            model=fit(train,weights)
            n=sum(equivalent(b,y,predict(model,b)) for b,y,g in test)
            folds.append({"fold":k,"n":len(test),"correct_text":n})
            print("WEIGHT_FOLD",name,k,n,flush=True)
        report["configs"][name]={"weights":list(weights),"folds":folds,"correct_text":sum(x["correct_text"] for x in folds)}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("WEIGHT_RESULTS_BEGIN");print(json.dumps(report,indent=2));print("WEIGHT_RESULTS_END")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True)
    p.add_argument("--output",default="research-results/vnext-family-weights")
    main(p.parse_args())
