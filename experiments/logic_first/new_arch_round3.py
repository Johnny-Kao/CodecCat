"""Independent new-architecture Round 3: three distinct evidence mechanisms.
No old Runtime, R21, old feature schema or H-series imports.
Competitor-owned data is DEV ONLY. One job, fixed groups, no hyperparameter search.
"""
import argparse,codecs,hashlib,json,re,resource,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

SEED=20261015
CANDIDATES=("byte_statistical","decoded_language","combined_evidence","logic_only")
LENGTHS=(128,1024,4096)
def canonical(v):
    try:return codecs.lookup(v).name.replace("_","-")
    except LookupError:return None
def corpus(root):
    import sys
    sys.path.insert(0,str(root/"scripts"))
    from check_test_data import parse_dir_name
    rows=[];dedup=set();reasons=Counter()
    for p in sorted(root.glob("*/*")):
        if not p.is_file():continue
        enc,language=parse_dir_name(p.parent.name)
        label=canonical(enc)
        if not label:reasons["unknown_label"]+=1;continue
        b=p.read_bytes()
        if not b:continue
        try:t=b.decode(enc,"strict")
        except (UnicodeDecodeError,LookupError):reasons["wrong_label"]+=1;continue
        digest=hashlib.sha256(b).hexdigest()
        if digest in dedup:reasons["duplicate"]+=1;continue
        dedup.add(digest)
        group=p.stem.lower().replace("culturax_mC4_","culturax_").replace("culturax_OSCAR-2301_","culturax_")
        fold=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%3
        rows.append({"path":str(p.relative_to(root)),"label":label,"language":language or "unknown",
                     "source":"transcoded" if p.name.startswith("culturax_") else "other",
                     "fold":fold,"group":group,"length":len(b),"data":b[:4096],
                     "ascii_only":all(x<128 for x in b[:4096])})
    return rows,dict(reasons)
def valid_decode(b,codec):
    for trim in (0,1,2,3):
        if trim and len(b)<=trim:break
        try:return (b[:-trim] if trim else b).decode(codec,"strict")
        except (UnicodeDecodeError,LookupError):pass
    return None
def text_shape(s):
    if s is None:return np.zeros(18,np.float32)
    n=max(1,len(s));u=np.fromiter((ord(c) for c in s),dtype=np.int32,count=len(s))
    if not len(u):return np.zeros(18,np.float32)
    intervals=((0x3040,0x309f),(0x30a0,0x30ff),(0x4e00,0x9fff),(0xac00,0xd7af),
               (0x0400,0x04ff),(0x0600,0x06ff),(0x0370,0x03ff),(0x0590,0x05ff),
               (0x0000,0x001f),(0x007f,0x009f),(0x0020,0x007e),(0x00c0,0x024f),
               (0x2000,0x206f),(0x0030,0x0039),(0x0100,0x017f),(0x0180,0x024f))
    v=[((u>=a)&(u<=z)).sum()/n for a,z in intervals]
    v.extend([min(n,4096)/4096,len(set(s))/n])
    return np.asarray(v,np.float32)
def train_language_refs(rows,indices):
    profiles=defaultdict(list)
    for i in indices:
        r=rows[i];decoded=valid_decode(r["data"],r["label"])
        if decoded is not None:profiles[r["label"]].append(text_shape(decoded))
    return {label:np.mean(v,axis=0) for label,v in profiles.items()}
def run(args):
    rows,excluded=corpus(Path(args.fixtures))
    assert len(rows)>1500
    labels=np.array([r["label"] for r in rows]); folds=np.array([r["fold"] for r in rows])
    variants={};traces=[];fit_seconds=Counter()
    for limit in LENGTHS:
        text=[r["data"][:limit].decode("latin1") for r in rows]
        vectorizer=HashingVectorizer(analyzer="char",ngram_range=(2,3),n_features=4096,
                     alternate_sign=False,norm="l1",lowercase=False,dtype=np.float32)
        X=vectorizer.transform(text)
        preds={k:np.empty(len(rows),dtype=object) for k in CANDIDATES}
        coverage=Counter();logic_counts=Counter();fold_details=[]
        for fold in (0,1,2):
            tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
            assert set(rows[i]["group"] for i in tr).isdisjoint({rows[i]["group"] for i in te})
            tic=time.perf_counter()
            clf=MultinomialNB(alpha=0.5).fit(X[tr],labels[tr])
            fit_seconds["byte_statistical"]+=time.perf_counter()-tic
            tic=time.perf_counter()
            refs=train_language_refs(rows,tr)
            fit_seconds["decoded_language"]+=time.perf_counter()-tic
            classes=list(clf.classes_)
            for i in te:
                b=rows[i]["data"][:limit]
                scores=clf.predict_log_proba(X[i])[0]
                ordering=np.argsort(scores)[::-1]
                raw=classes[ordering[0]]
                dec_results={c:valid_decode(b,c) for c in classes}
                valid=[c for c in classes if dec_results[c] is not None]
                coverage["candidate_count_total"]+=len(valid)
                coverage["truth_valid"]+=int(rows[i]["label"] in valid)
                if len(valid)>1:coverage["ambiguous"]+=1
                if not valid:coverage["none_valid"]+=1
                # Decoded text script profile distance from SOURCE-DISJOINT fitted labels.
                langdist={c:float(np.abs(text_shape(dec_results[c])-refs[c]).sum())
                          for c in valid if c in refs}
                langchoice=min(langdist,key=langdist.get) if langdist else raw
                valid_order=[c for c in (classes[j] for j in ordering) if c in valid]
                bytechoice=valid_order[0] if valid_order else raw
                # Same-fit, no hardcoded learned weights; rank product uses normalized
                # byte relative log-probability and script distance.
                if langdist:
                    top=max(scores)
                    blend={c:float(scores[classes.index(c)]-top)-8*langdist[c]
                           for c in langdist}
                    combo=max(blend,key=blend.get)
                else:combo=raw
                preds["byte_statistical"][i]=bytechoice
                preds["decoded_language"][i]=langchoice
                preds["combined_evidence"][i]=combo
                preds["logic_only"][i]=valid[0] if len(valid)==1 else None
                if limit==4096:
                    traces.append({"path":rows[i]["path"],"fold":fold,"label":rows[i]["label"],
                                  "source":rows[i]["source"],"language":rows[i]["language"],
                                  "valid_count":len(valid),
                                  "predictions":{k:str(preds[k][i]) for k in CANDIDATES}})
            fold_details.append({"fold":fold,"train":len(tr),"test":len(te),
                "unseen_labels":sum(labels[i] not in set(labels[tr]) for i in te)})
        scored={}
        for k,p in preds.items():
            hits=(p==labels)
            scored[k]={"n":len(rows),"hits":int(hits.sum()),"accuracy":round(float(hits.mean()),5),
                       "by_source":{v:{"n":sum(r["source"]==v for r in rows),
                          "hits":int(sum(hits[i] for i,r in enumerate(rows) if r["source"]==v))}
                          for v in ("transcoded","other")},
                       "by_encoding":{v:{"n":int(sum(labels==v)),"hits":int(sum(hits[labels==v]))}
                          for v in sorted(set(labels))}}
        variants[str(limit)]={"scores":scored,"coverage":dict(coverage),"folds":fold_details}
        print("ROUND3_LENGTH_COMPLETE",limit,flush=True)
    # Runtime measured separately from training and includes decoding evidence.
    t0=time.perf_counter()
    for r in rows[:500]:
        for c in ("utf-8","shift-jis","euc-jp","big5","gb18030"):
            valid_decode(r["data"],c)
    decoder_ns=int((time.perf_counter()-t0)*1e9/500)
    report={"round":3,"type":"full independent evidence topology comparison",
      "source":"chardet/test-data — development-only, competitor-owned",
      "n":len(rows),"excluded":excluded,"seed":SEED,"schema":"separate directional statistical vs decoded-script coherence v1",
      "variants":variants,"fit_seconds":{k:round(v,3) for k,v in fit_seconds.items()},
      "five_decoder_ns_per_sample":decoder_ns,
      "peak_mixed_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
      "warnings":["Scores on competitor's own fixtures are NOT independent acceptance",
                  "Decoder scan measured separately; full end-to-end per-model p95 still required",
                  "Script profile uses source-disjoint fitting but not verified native language labels",
                  "No R21 imports, artifacts, features or controls."]}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"trace.jsonl").write_text("\n".join(json.dumps(x) for x in traces)+"\n")
    print("ROUND3_RESULTS_BEGIN")
    print(json.dumps({"n":len(rows),"excluded":excluded,
        "summaries":{k:{"scores":{c:{"hits":z["hits"],"accuracy":z["accuracy"]} for c,z in v["scores"].items()},
                        "coverage":v["coverage"]} for k,v in variants.items()},
        "fit_seconds":report["fit_seconds"],"five_decoder_ns_per_sample":decoder_ns,
        "peak_mixed_rss_mib":report["peak_mixed_rss_mib"]},indent=2))
    print("ROUND3_RESULTS_END")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True)
    p.add_argument("--output",default="research-results/new-arch-round3")
    run(p.parse_args())
