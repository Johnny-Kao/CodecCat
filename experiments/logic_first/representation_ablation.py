"""Representation ablation for group-held-out mixed-source development panels.

No modification to production runtime. All candidates keep identical L1, L3,
training splits and input bytes; only L2's feature representation differs.
This is development evidence, not a fresh independent holdout.
"""
from collections import Counter,defaultdict
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from five_panel_tournament import make_cases,distribute,strict,SEED

PROFILES={
    "binary13":((1,3),True,1,90000),
    "count12":((1,2),False,2,60000),
}
def train_candidate(train,name):
    grams,binary,min_df,max_features=PROFILES[name]
    vec=TfidfVectorizer(analyzer="char",ngram_range=grams,binary=binary,
        min_df=min_df,max_features=max_features,sublinear_tf=True)
    x=vec.fit_transform([b[:4096].decode("latin1") for b,y,g in train])
    clf=SGDClassifier(loss="modified_huber",alpha=.0003,random_state=20261015,max_iter=1500)
    clf.fit(x,[y for b,y,g in train])
    return TrainableDetector(vectorizer=vec,classifier=clf,fingerprint="representation-ablation-"+name)

def evaluate_folds(rows,fold):
    train=[(b,y,g) for b,y,g,k,sha in rows if k!=fold] if fold!=-1 else [(b,y,g) for b,y,g,k,sha in rows]
    residual=residual_subset(train)[0]
    language=LanguageCoherence.fit(train)
    models={"baseline":fit(residual,"binary_high")}
    models.update({name:train_candidate(residual,name) for name in PROFILES})
    return models,language

def main():
    rows=corpus(Path("training"))
    pool,_=make_cases(rows);panels=distribute(pool)
    training={fold:evaluate_folds(rows,fold) for fold in (0,1,2,-1)}
    by_panel=defaultdict(lambda:defaultdict(Counter))
    by_source=defaultdict(lambda:defaultdict(Counter))
    by_size=defaultdict(lambda:defaultdict(Counter))
    totals=defaultdict(Counter)
    cases=[]
    for number,panel in enumerate(panels,1):
        for row in panel:
            expected=strict(row["raw"],row["label"])
            models,language=training[row["fold"]]
            outcome={}
            for name,model in models.items():
                r=detect(model,row["raw"],language_model=language)
                ok=r.accepted and r.text==expected
                outcome[name]={"encoding":r.encoding,"correct":bool(ok),"reason":r.reason}
                for counter in (totals[name],by_panel[number][name],by_source[row["source"]][name],
                                by_size["short<=256" if len(row["raw"])<=256 else
                                        "medium<=4096" if len(row["raw"])<=4096 else "large>4096"][name]):
                    counter["n"]+=1;counter["correct"]+=int(ok)
            base=outcome["baseline"]["correct"]
            for name in PROFILES:
                if outcome[name]["correct"] and not base:totals[name]["fixes"]+=1
                if base and not outcome[name]["correct"]:totals[name]["breaks"]+=1
            cases.append({"sha":row["sha"],"panel":number,"source":row["source"],
                "truth":row["label"],"bytes":len(row["raw"]),"outcome":outcome})
        print("REPRESENTATION_PANEL_DONE",number,flush=True)
    def flatten(x):
        return {k:{name:{**dict(c),"accuracy":round(c["correct"]/c["n"],5)}
                   for name,c in v.items()} for k,v in x.items()}
    report={"seed":SEED,
        "totals":{name:{**dict(c),"accuracy":round(c["correct"]/c["n"],5)}
                  for name,c in totals.items()},
        "by_panel":flatten(by_panel),"by_source":flatten(by_source),
        "by_length":flatten(by_size),
        "limitations":["Repeatedly used source fixtures; only group-held-out training, not independent confirmation.",
          "Models differ in features only; same fixed layer routing and five panels.",
          "This study compares two preregistered representation variants and current research baseline."]}
    out=Path("research-results/representation-ablation");out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases)+"\n")
    print("REPRESENTATION_ABLATION_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
