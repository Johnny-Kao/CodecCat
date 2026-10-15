"""New-architecture data/model gate: grouped validation, independent of old runtime.

This is development comparison, NOT a release qualification. No R21 imports.
"""
from __future__ import annotations
import codecs,hashlib,json,statistics,time,argparse,resource
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

def canonical(enc):
    try:return codecs.lookup(enc).name
    except LookupError:return None

def collect(path):
    import sys
    sys.path.insert(0,str(path/"scripts"))
    from check_test_data import parse_dir_name
    out=[];dedup=set()
    for f in sorted(path.glob("*/*")):
        if not f.is_file():continue
        enc,_=parse_dir_name(f.parent.name);label=canonical(enc)
        if not label:continue
        b=f.read_bytes()
        if not b:continue
        try:b.decode(label,"strict")
        except (UnicodeDecodeError,LookupError):continue
        sha=hashlib.sha256(b).hexdigest()
        if sha in dedup:continue
        dedup.add(sha)
        # Preserve entire parent stem across re-encodings, group correlated texts.
        key=f.stem.lower().replace("culturax_mc4_","culturax_").replace("culturax_oscar-2301_","culturax_")
        fold=int(hashlib.sha256(key.encode()).hexdigest()[:12],16)%3
        out.append((b[:4096],label,key,fold,sha))
    return out

def safe_survivors(data,classes,complete):
    # Full strict decoder on complete bytes, prefix incremental on truncated.
    survivors=[]
    for label in classes:
        try:
            dec=codecs.getincrementaldecoder(label)("strict")
            dec.decode(data,final=complete)
            survivors.append(label)
        except (UnicodeDecodeError,LookupError):pass
    return survivors

def run(a):
    rows=collect(Path(a.fixtures));n=len(rows)
    if n<1500:raise RuntimeError("small corpus")
    truth=np.array([r[1] for r in rows])
    folds=np.array([r[3] for r in rows])
    text=[r[0].decode("latin1") for r in rows]
    # Different evidence/decision topologies, not repeated smoothing experiments.
    families={
       "byte12_count_comp":(CountVectorizer(analyzer="char",ngram_range=(1,2),max_features=28000),ComplementNB(alpha=0.1)),
       "ordered23_tfidf_svm":(TfidfVectorizer(analyzer="char",ngram_range=(2,3),min_df=2,max_features=45000,sublinear_tf=True),LinearSVC(C=0.5)),
       "ordered23_tfidf_logistic":(TfidfVectorizer(analyzer="char",ngram_range=(2,3),min_df=2,max_features=45000,sublinear_tf=True),SGDClassifier(loss="modified_huber",alpha=0.0001,random_state=20261015,max_iter=1500))
    }
    outputs={k:[None]*n for k in families}
    timings={}
    per_fold=[]
    for hold in range(3):
        train=np.flatnonzero(folds!=hold)
        test=np.flatnonzero(folds==hold)
        assert {rows[i][2] for i in train}.isdisjoint({rows[i][2] for i in test})
        per_fold.append({"fold":hold,"train":len(train),"test":len(test),
                         "unseen_truth":int(sum(truth[i] not in set(truth[train]) for i in test))})
        for name,(vectorizer,model) in families.items():
            t=time.perf_counter()
            Xtr=vectorizer.fit_transform([text[i] for i in train])
            model.fit(Xtr,truth[train])
            fit_s=time.perf_counter()-t
            t=time.perf_counter()
            Xte=vectorizer.transform([text[i] for i in test])
            scores=model.decision_function(Xte) if hasattr(model,"decision_function") else model.predict_log_proba(Xte)
            if scores.ndim==1:scores=np.column_stack([-scores,scores])
            ordering=np.argsort(scores,axis=1)[:,::-1]
            labels=model.classes_
            classify_s=time.perf_counter()-t
            for k,i in enumerate(test):
                ranked=[str(labels[j]) for j in ordering[k]]
                candidates=safe_survivors(rows[i][0],ranked,complete=(len(rows[i][0])<4096))
                filtered=next((c for c in ranked if c in candidates),None)
                raw=ranked[0]
                # L3 information distinct from validity: 4k decoded text control penalty
                top=candidates[:2]
                best=filtered
                if len(top)==2:
                    penalties={}
                    for label in top:
                        try:
                            decoded=rows[i][0].decode(label,"strict")
                            ct=sum((ord(c)<32 and c not in "\t\r\n") or 127<=ord(c)<160 for c in decoded)
                            penalties[label]=ct/max(len(decoded),1)
                        except (UnicodeDecodeError,LookupError):penalties[label]=1
                    if penalties[top[1]]+0.02<penalties[top[0]]:
                        best=top[1]
                outputs[name][i]={"raw":raw,"l1":filtered,"full":best,
                     "retained":str(truth[i]) in candidates,"n_survivors":len(candidates),
                     "l3_changed":best!=filtered,"top5":ranked[:5]}
            timings.setdefault(name,[]).append({"fit_s":round(fit_s,3),"score_s":round(classify_s,3)})
        print("FOLD_DONE",hold,flush=True)
    report={"source":"chardet/test-data, competitor-owned, development-only","n":n,
            "folds":per_fold,"models":{},"timings":timings,
            "mixed_process_peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2)}
    for name,preds in outputs.items():
        truth_list=[str(x) for x in truth]
        report["models"][name]={
            "raw_hits":sum(p["raw"]==y for p,y in zip(preds,truth_list)),
            "l1_hits":sum(p["l1"]==y for p,y in zip(preds,truth_list)),
            "full_hits":sum(p["full"]==y for p,y in zip(preds,truth_list)),
            "truth_excluded":sum(not p["retained"] for p in preds),
            "l3_changes":sum(p["l3_changed"] for p in preds),
            "l3_helps":sum(p["full"]==y and p["l1"]!=y for p,y in zip(preds,truth_list)),
            "l3_harms":sum(p["full"]!=y and p["l1"]==y for p,y in zip(preds,truth_list)),
            "top5_hits":sum(y in p["top5"] for p,y in zip(preds,truth_list))
        }
    path=Path(a.output);path.mkdir(parents=True,exist_ok=True)
    (path/"report.json").write_text(json.dumps(report,indent=2))
    (path/"trace.jsonl").write_text("\n".join(json.dumps({"sha":row[4],"truth":row[1],"source_group":row[2],"fold":row[3],
           "models":{k:outputs[k][i] for k in outputs}}) for i,row in enumerate(rows))+"\n")
    print("RESULT_BEGIN");print(json.dumps(report,indent=2));print("RESULT_END")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fixtures",required=True)
    p.add_argument("--output",default="research-results/vnext-strong-model-dev")
    run(p.parse_args())
