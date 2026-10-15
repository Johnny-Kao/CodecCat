"""Strict 3-fold source-group L2 feature ablation. No external set used.

Compare candidate recall and decoded-text quality; training rows never share
a source-group with their evaluation fold.
"""
from __future__ import annotations
import argparse,collections,json,sys,codecs
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from codeccat.vnext_strong_l2 import TrainableDetector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus

CONFIGS={
 "baseline":dict(ngram_range=(2,3),min_df=2,max_features=45000,alpha=0.0001,class_weight=None),
 "coverage_24":dict(ngram_range=(2,4),min_df=1,max_features=90000,alpha=0.0001,class_weight=None),
 "coverage_13":dict(ngram_range=(1,3),min_df=1,max_features=90000,alpha=0.0001,class_weight=None),
 "balanced_23":dict(ngram_range=(2,3),min_df=2,max_features=45000,alpha=0.0001,class_weight="balanced")
}
def fit(rows,cfg):
    X=[x[:4096].decode("latin1") for x,y,g in rows]
    y=[label for x,label,g in rows]
    vec=TfidfVectorizer(analyzer="char",ngram_range=cfg["ngram_range"],min_df=cfg["min_df"],
                        max_features=cfg["max_features"],sublinear_tf=True)
    tx=vec.fit_transform(X)
    clf=SGDClassifier(loss="modified_huber",alpha=cfg["alpha"],max_iter=1500,
                      random_state=20261015,class_weight=cfg["class_weight"])
    clf.fit(tx,y)
    return TrainableDetector(vectorizer=vec,classifier=clf,fingerprint="ablation")
def evaluate(m,rows):
    c=collections.Counter()
    for raw,truth,g in rows:
        ranking=m.rank(raw,final=len(raw)<4096)
        guess=ranking.encoding
        try:
            target=codecs.getincrementaldecoder(truth)("strict").decode(raw,final=len(raw)<4096)
            pred=codecs.getincrementaldecoder(guess)("strict").decode(raw,final=len(raw)<4096) if guess else None
            same=pred==target
        except (LookupError,UnicodeError,TypeError):same=False
        c["n"]+=1
        c["exact"]+=int(guess==truth)
        c["text"]+=int(same)
        c["top3"]+=int(truth in ranking.candidates)
    return dict(c)
def main(a):
    rows=corpus(Path(a.train))
    out={"n":len(rows),"source_groups":len({x[2] for x in rows}),
         "warning":"source-group dev comparison only; do not tune on external holdouts",
         "configs":{}}
    for key,cfg in CONFIGS.items():
        folds=[]
        for k in range(3):
            tr=[(b,y,g) for b,y,g,fold,h in rows if fold!=k]
            te=[(b,y,g) for b,y,g,fold,h in rows if fold==k]
            assert {r[2] for r in tr}.isdisjoint({r[2] for r in te})
            m=fit(tr,cfg)
            folds.append(evaluate(m,te))
            print("ABLATION_FOLD_DONE",key,k,flush=True)
        c=collections.Counter()
        for f in folds:c.update(f)
        out["configs"][key]={"folds":folds,"aggregate":dict(c)}
    p=Path(a.output);p.mkdir(parents=True,exist_ok=True)
    (p/"report.json").write_text(json.dumps(out,indent=2))
    print("FEATURE_ABLATION_BEGIN");print(json.dumps(out,indent=2));print("FEATURE_ABLATION_END")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--train",required=True)
    p.add_argument("--output",default="research-results/vnext-l2-feature-ablation")
    main(p.parse_args())
