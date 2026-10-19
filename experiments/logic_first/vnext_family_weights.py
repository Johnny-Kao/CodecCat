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
class WeightedFeatures:
    def __init__(self,vectorizer,factors):
        self.vectorizer=vectorizer
        self.factors=factors
    def transform(self,values):
        from sklearn.preprocessing import normalize
        return normalize(self.vectorizer.transform(values).multiply(self.factors).tocsr())

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
    return TrainableDetector(vectorizer=WeightedFeatures(vec,factors),classifier=clf,fingerprint="weight-research")
def predict(model,b):
    return model.rank(b,final=len(b)<4096).encoding
def equivalent(b,y,p):
    try:return codecs.getincrementaldecoder(y)("strict").decode(b,final=len(b)<4096)==codecs.getincrementaldecoder(p)("strict").decode(b,final=len(b)<4096)
    except (LookupError,UnicodeError,TypeError):return False
def main(args):
    rows=corpus(Path(args.fixtures))
    report={"n":len(rows),"configs":{},"warning":"group-disjoint dev-only experiment on full production L1+L2 rank path; do not promote without holdout and numeric archive schema"}
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
