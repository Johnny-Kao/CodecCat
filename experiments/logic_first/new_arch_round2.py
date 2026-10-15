"""New architecture round 2: independent, grouped, multi-topology evidence tournament.

Pure new research. No import/download/use of previous R21/H-series.
Test set is competitor-owned chardet/test-data: exploratory only.
"""
import argparse,codecs,hashlib,json,resource,time,statistics
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB,ComplementNB
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack,csr_matrix

SEED=20261015
CONFIGS={
 "ngram_nb":("n23","nb"),
 "ngram_complement":("n23","comp"),
 "ngram_sgd":("n23","sgd"),
 "trigram_sgd":("n3","sgd"),
 "hybrid_sgd":("hybrid","sgd"),
 "hybrid_comp":("hybrid","comp"),
}
def canon(name):
    try:return codecs.lookup(name).name.replace("_","-")
    except LookupError:return None
def load(root):
    import sys
    sys.path.insert(0,str(root/"scripts"))
    from check_test_data import parse_dir_name
    rows=[];seen=set();skipped=Counter()
    for p in sorted(root.glob("*/*")):
        if not p.is_file():continue
        enc,lang=parse_dir_name(p.parent.name)
        lab=canon(enc)
        if not lab:skipped["unknown"]+=1;continue
        data=p.read_bytes()
        if not data:skipped["empty"]+=1;continue
        try:data.decode(enc,"strict")
        except (UnicodeDecodeError,LookupError):skipped["invalid_label"]+=1;continue
        key=hashlib.sha256(data).hexdigest()
        if key in seen:skipped["exact_duplicate"]+=1;continue
        seen.add(key)
        # hash suffix groups to keep derivatives from same source together
        group=p.stem.lower().replace("culturax_mC4_","culturax_").replace("culturax_OSCAR-2301_","culturax_")
        fold=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%3
        rows.append({"path":p.relative_to(root).as_posix(),"group":group,
                     "fold":fold,"language":lang or "?","label":lab,
                     "source":"transcoded" if p.name.startswith("culturax_") else "other",
                     "bytes":data[:4096],"full_length":len(data)})
    return rows,skipped

def evidence(rows,cut):
    data=[r["bytes"][:cut] for r in rows]
    strings=[d.decode("latin1") for d in data]
    result={}
    # Hashing fixed feature IDs; no fitting on test fold.
    for key,ngram in [("n23",(2,3)),("n3",(3,3)),("single",(1,1))]:
        vec=HashingVectorizer(analyzer="char",ngram_range=ngram,n_features=8192 if key!="single" else 256,
                              alternate_sign=False,norm="l1",lowercase=False,dtype=np.float32)
        result[key]=vec.transform(strings)
    result["hybrid"]=hstack([result["n23"],result["single"]],format="csr")
    return result

def valid_prefix(data,codec):
    try: codecs.lookup(codec)
    except LookupError:return None
    for trim in range(4):
        if trim and len(data)<=trim:break
        try:
            (data[:-trim] if trim else data).decode(codec,"strict")
            return True
        except UnicodeDecodeError:pass
    return False

def structural_policy(ranking,data,apply):
    if not apply:return ranking[0]
    for candidate in ranking[:5]:
        v=valid_prefix(data,candidate)
        if v is not False:return candidate
    return ranking[0]

def fit_model(kind):
    if kind=="nb":return MultinomialNB(alpha=0.1)
    if kind=="comp":return ComplementNB(alpha=0.5)
    return SGDClassifier(loss="log_loss",alpha=1e-4,class_weight="balanced",
                         random_state=SEED,max_iter=1600,tol=1e-4)

def tournament(rows,cut,only_models=None):
    X=evidence(rows,cut);y=np.array([r["label"] for r in rows])
    out={k:[] for k in CONFIGS};cost=Counter();per_fold=[]
    for hold in range(3):
        tr=np.flatnonzero([r["fold"]!=hold for r in rows]);te=np.flatnonzero([r["fold"]==hold for r in rows])
        assert set(rows[i]["group"] for i in tr).isdisjoint(set(rows[i]["group"] for i in te))
        per_fold.append({"fold":hold,"train":len(tr),"test":len(te),"unseen_labels":int(sum(y[i] not in set(y[tr]) for i in te))})
        for name,(feat,kind) in CONFIGS.items():
            if only_models and name not in only_models:continue
            start=time.perf_counter()
            clf=fit_model(kind).fit(X[feat][tr],y[tr])
            costs=time.perf_counter()-start;cost[name]+=costs
            # rank candidates by log posterior, including when classes absent from train.
            logits=clf.predict_log_proba(X[feat][te]) if kind!="sgd" else clf.predict_proba(X[feat][te])
            order=np.argsort(logits,axis=1)[:,::-1]
            classes=clf.classes_
            for idx,row_order in zip(te,order):
                ranked=[str(classes[j]) for j in row_order]
                raw=ranked[0]
                safe=structural_policy(ranked,rows[idx]["bytes"][:cut],True)
                top2=float(logits[list(te).index(idx),row_order[0]]-logits[list(te).index(idx),row_order[1]]) if len(row_order)>1 else 0.0
                out[name].append((idx,raw,safe,top2,ranked[:5]))
    summarized={}
    for name,records in out.items():
        if not records:continue
        truth=lambda idx:rows[idx]["label"]
        summary={"n":len(records),"raw_hits":sum(pred==truth(i) for i,pred,_,_,_ in records),
                 "logic_hits":sum(pred==truth(i) for i,_,pred,_,_ in records),
                 "top5_hits":sum(truth(i) in ranking for i,_,_,_,ranking in records),
                 "logic_changes":sum(a!=b for _,a,b,_,_ in records),
                 "grouped_training_sec":round(cost[name],3)}
        summarized[name]=summary
    return summarized,out,per_fold

def main(args):
    rows,skip=load(Path(args.fixtures))
    if len(rows)<1500:raise RuntimeError(f"Insufficient training data: {len(rows)}")
    result={"run":"independent new architecture round2","n":len(rows),
            "source":"chardet/test-data competitor-owned development only",
            "exclusions":dict(skip),"candidates":{},"strata":{},
            "settings":{"seed":SEED,"folds":3,"features":"directional 2/3-char byte ngrams hashed 8192, sparse",
                        "policies":"raw versus lazy top-five strict decoder evidence"}}
    traces=[]
    start=time.perf_counter()
    for cut in (128,1024,4096):
        summary,records,folds=tournament(rows,cut)
        result["candidates"][str(cut)]=summary
        result["folds"]=folds
        for name,seq in records.items():
            if cut==4096:
                for i,pred,safe,margin,rank in seq:
                    traces.append({"sha":hashlib.sha256(rows[i]["bytes"]).hexdigest(),
                                   "fold":rows[i]["fold"],"source":rows[i]["source"],
                                   "truth":rows[i]["label"],"language":rows[i]["language"],
                                   "model":name,"prediction":pred,"logic_prediction":safe,
                                   "ranking":rank})
        print("ROUND2_CUT_DONE",cut,flush=True)
    result["training_evaluation_s"]=round(time.perf_counter()-start,3)
    result["process_peak_rss_mib"]=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2)
    result["best_full_length"]=sorted([(k,v["logic_hits"],v["raw_hits"]) for k,v in result["candidates"]["4096"].items()],key=lambda z:z[1],reverse=True)
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    (dest/"report.json").write_text(json.dumps(result,indent=2))
    (dest/"trace.jsonl").write_text("\n".join(json.dumps(x) for x in traces)+"\n")
    print("NEW_ROUND2_RESULT_START")
    print(json.dumps({k:v for k,v in result.items() if k not in ("strata",)},indent=2))
    print("NEW_ROUND2_RESULT_END")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True)
    p.add_argument("--output",default="research-results/new-arch-round2")
    main(p.parse_args())
