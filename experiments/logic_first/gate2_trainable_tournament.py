"""Gate 2/5: same-source grouped cross-year parameterized feature tournament.

Development-only, previous crawls, declared/strict labels not verified holdout.
All models use one fixed SGD configuration; no threshold search.
"""
import argparse,gzip,hashlib,json,random,resource,statistics,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import chardet,charset_normalizer
from charset_normalizer import from_bytes
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from codeccat import load_bundle
from codeccat.runtime import Runtime,family
from codeccat.vnext_features import SCHEMA,feature_variants
from h2_fused_candidate_mask_ab import collect,normalize,base as collector_module
from h14_japanese_fastpath import fast as h14

SHA="e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54"
CRAWLS=("CC-MAIN-2024-46","CC-MAIN-2025-13","CC-MAIN-2025-51",
        "CC-MAIN-2026-08","CC-MAIN-2026-12","CC-MAIN-2026-39")
MODELS=("base","base_logic","base_ordered","combined")
SEED=20261015
CONFIDENCE=0.90

def main(a):
    if hashlib.sha256(Path(a.model).read_bytes()).hexdigest()!=SHA:raise RuntimeError("R21 SHA mismatch")
    rt=Runtime(load_bundle(a.model));seen=set();records=[];feature_parts=defaultdict(list)
    times=Counter()
    for crawl in CRAWLS:
        samples,_,_=collect(crawl);start_count=len(records)
        for s in samples:
            data=s["data"];digest=hashlib.sha256(data).hexdigest()
            if digest in seen:continue
            seen.add(digest)
            label=normalize(s["label"]);yr=int(crawl.split("-")[2])
            p=rt.rank(data)
            ctrl=h14(data,p[0])[0]
            # Non-controversial strict guard from Gate 1 is imported locally.
            from gate1_architecture_tournament import late
            ctrl=late(data,(ctrl,)+tuple(c for c in p if c!=ctrl),"strict")
            declared=collector_module.declared_body_label(data) is not None
            tic=time.perf_counter_ns()
            f=feature_variants(data)
            times["extract_features_ns"]+=time.perf_counter_ns()-tic
            for k,v in f.items():feature_parts[k].append(v)
            records.append({"crawl":crawl,"year":yr,"host":str(s.get("host") or ""),
                "warc_path":str(s.get("warc_path") or ""),"tier":str(s.get("tier") or ""),
                "sha":digest,"truth":label,"family":family(label),"visible_declaration":declared,
                "r21":p[0],"h14_strict":ctrl,"bytes":len(data)})
        print("GATE2_COLLECTED",crawl,len(records)-start_count,flush=True)
    if len(records)<500:raise RuntimeError("insufficient dataset")
    X={k:np.stack(v).astype(np.float32) for k,v in feature_parts.items()}
    Y=np.array([r["truth"] for r in records],dtype=object)
    folds={};oof=defaultdict(dict);fit_sec=Counter()
    for hold_year in (2024,2025,2026):
        test=np.array([i for i,r in enumerate(records) if r["year"]==hold_year],dtype=np.intp)
        # No host from the evaluation set is allowed in this fold's training.
        test_hosts={records[i]["host"] for i in test if records[i]["host"]}
        train=np.array([i for i,r in enumerate(records) if r["year"]!=hold_year
                        and (not r["host"] or r["host"] not in test_hosts)],dtype=np.intp)
        fold={"train_n":len(train),"test_n":len(test),"train_classes":sorted(set(Y[train])),
            "eval_unseen_labels":dict(Counter(str(Y[i]) for i in test if Y[i] not in set(Y[train]))),
            "host_overlap":len(test_hosts.intersection({records[i]["host"] for i in train}))}
        assert len(train)>=100 and len(test)>=100 and fold["host_overlap"]==0
        for model_name in MODELS:
            tic=time.perf_counter()
            scaler=StandardScaler()
            a_train=scaler.fit_transform(X[model_name][train])
            a_test=scaler.transform(X[model_name][test])
            clf=SGDClassifier(loss="log_loss",alpha=0.0001,max_iter=1200,
                tol=0.0005,class_weight="balanced",random_state=SEED)
            clf.fit(a_train,Y[train])
            pred=clf.predict(a_test)
            prob=clf.predict_proba(a_test)
            fit_sec[model_name]+=time.perf_counter()-tic
            classes=list(clf.classes_)
            for idx,raw,probs in zip(test,pred,prob):
                score=float(max(probs))
                # Honest low-cost policy: uncertain learned guesses abstain.
                hybrid=raw if score>=CONFIDENCE else records[idx]["h14_strict"]
                oof[int(idx)][model_name]={"direct":str(raw),
                    "hybrid":str(hybrid),"confidence":round(score,6)}
        folds[str(hold_year)]=fold
        print("GATE2_TRAINED_HELDOUT",hold_year,fold["train_n"],fold["test_n"],flush=True)
    def count(which,subset):
        if which in ("r21","h14_strict"):
            pred=lambda i:records[i][which]
        elif which.startswith("direct_"):
            m=which[7:];pred=lambda i:oof[i][m]["direct"]
        else:
            m=which[7:];pred=lambda i:oof[i][m]["hybrid"]
        hits=sum(pred(i)==records[i]["truth"] for i in subset)
        old=sum(records[i]["h14_strict"]==records[i]["truth"] for i in subset)
        return {"n":len(subset),"hits":hits,"accuracy":round(hits/max(1,len(subset)),6),
                "delta_vs_h14_strict":hits-old,
                "harmed":sum(pred(i)!=records[i]["truth"] and records[i]["h14_strict"]==records[i]["truth"] for i in subset),
                "helped":sum(pred(i)==records[i]["truth"] and records[i]["h14_strict"]!=records[i]["truth"] for i in subset)}
    keys=["r21","h14_strict"]+[f"{p}_{m}" for p in ("direct","hybrid") for m in MODELS]
    all_ix=list(range(len(records)))
    overall={k:count(k,all_ix) for k in keys}
    splits={}
    for key in ("year","family","visible_declaration"):
        groups={}
        for value in sorted(set(r[key] for r in records)):
            idx=[i for i,r in enumerate(records) if r[key]==value]
            groups[str(value)]={k:count(k,idx) for k in keys}
        splits[key]=groups
    # Cheap extractor throughput separate from scorer (these are NOT end-to-end).
    rng=random.Random(SEED);subset=rng.sample(all_ix,min(384,len(records)))
    # Persist model-free fixed features to avoid refetch when only downstream scoring changes.
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"development_feature_cache.npz",
        **{k:X[k] for k in MODELS},labels=Y.astype(str),
        shas=np.array([r["sha"] for r in records]),years=np.array([r["year"] for r in records]))
    report={"gate":2,"feature_schema":SCHEMA,"training_config":{"model":"SGDClassifier",
         "loss":"log_loss","alpha":0.0001,"max_iter":1200,"class_weight":"balanced",
         "seed":SEED,"hybrid_confidence_threshold":CONFIDENCE},
         "data_warning":"cross-year GROUPED DEVELOPMENT validation; previously research-exposed crawls, not independently verified labels",
         "n":len(records),"collection":dict(Counter(r["crawl"] for r in records)),
         "label_provenance":dict(Counter(r["tier"] for r in records)),
         "visible_declaration_n":sum(r["visible_declaration"] for r in records),
         "folds":folds,"overall":overall,"splits":splits,
         "model_fit_seconds":{k:round(v,3) for k,v in fit_sec.items()},
         "feature_extraction_mean_ms":round(times["extract_features_ns"]/len(records)/1e6,4),
         "mixed_process_peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
         "notice":"No production model bundle emitted; feature schema incompatible with frozen R21 518-d scorer."}
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"trace.jsonl").write_text("\n".join(json.dumps({**r,"candidate":oof[i]}) for i,r in enumerate(records))+"\n")
    print("GATE2_RESULT_START",flush=True)
    print(json.dumps({key:report[key] for key in ("n","feature_schema","label_provenance","visible_declaration_n","folds","overall","model_fit_seconds","feature_extraction_mean_ms")},indent=2),flush=True)
    print("GATE2_RESULT_END",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",required=True)
    p.add_argument("--out",default="research-results/vnext-gate2")
    a=p.parse_args();main(a)
