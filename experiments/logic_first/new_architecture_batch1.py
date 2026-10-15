"""Independent Logic-First architecture tournament. NO R21/HMT/legacy imports.
Development corpus belongs to chardet; not independent release acceptance.
"""
import argparse,codecs,hashlib,json,resource,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.naive_bayes import MultinomialNB

SEED=20261015
BINS=1024
def canonical(name):
    try:return codecs.lookup(name).name.replace("_","-")
    except LookupError:return None

def features(data):
    b=np.frombuffer(data[:4096],dtype=np.uint8)
    u=np.bincount(b,minlength=256).astype(np.float32)
    pairs=np.zeros(BINS,dtype=np.float32)
    if len(b)>1:
        a=b[:-1].astype(np.uint32);z=b[1:].astype(np.uint32)
        ids=((a*257+z)*2654435761 & 0xffffffff)%BINS
        pairs=np.bincount(ids.astype(np.intp),minlength=BINS).astype(np.float32)
    return u,pairs

def evidence(data,codec):
    # Logic-first: strict C-backed decoder; no guessed label from bare validity.
    if codec is None:return 0
    s=data[:4096]
    for trim in (0,1,2,3):
        if trim and len(s)<=trim:break
        try:
            (s[:-trim] if trim else s).decode(codec,"strict")
            return 1
        except (UnicodeDecodeError,LookupError):continue
    return 0

def dataset(root):
    from check_test_data import parse_dir_name
    rows=[];reasons=Counter();duplicates=set()
    for path in sorted(root.glob("*/*")):
        if not path.is_file():continue
        enc,lang=parse_dir_name(path.parent.name)
        label=canonical(enc)
        if not label:reasons["unknown_label"]+=1;continue
        b=path.read_bytes()
        if not b:reasons["empty"]+=1;continue
        try:b.decode(enc,"strict")
        except (UnicodeDecodeError,LookupError):reasons["strict_invalid"]+=1;continue
        sig=hashlib.sha256(b).hexdigest()
        if sig in duplicates:reasons["duplicate_bytes"]+=1;continue
        duplicates.add(sig)
        # Related transcoded texts have same basename; keep in same source group.
        group=path.stem.lower()
        source="culturax" if path.name.startswith("culturax_") else "other"
        rows.append((path.relative_to(root).as_posix(),group,source,label,b[:4096]))
    return rows,dict(reasons)

def main(args):
    rows,excluded=dataset(Path(args.fixtures))
    if len(rows)<1500:raise RuntimeError("insufficient fixtures")
    # Assign every group to exactly one fold. In particular transcoded copies
    # from the same original text never leak between train and test.
    folds=np.array([int(hashlib.sha256(r[1].encode()).hexdigest()[:8],16)%3 for r in rows])
    labels=np.array([r[3] for r in rows])
    us=[];ps=[]
    tic=time.perf_counter()
    for r in rows:
        u,p=features(r[4]);us.append(u);ps.append(p)
    extraction_s=time.perf_counter()-tic
    us=np.stack(us);ps=np.stack(ps)
    data_sets={"unigram":us,"ordered_transition":ps,"unigram_plus_ordered":np.concatenate([us,ps],axis=1)}
    configurations=[(mode,alpha) for mode in data_sets for alpha in (0.1,1.0)]
    predictions={name:np.empty(len(rows),dtype=object) for name,_ in configurations for name in [f"{name}_a{_}"]}
    seen_label=np.zeros(len(rows),bool)
    fold_summaries=[];training_s=Counter()
    for fold in range(3):
        train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
        train_labels=set(labels[train]);seen_label[test]=np.array([x in train_labels for x in labels[test]])
        assert set(rows[i][1] for i in train).isdisjoint(set(rows[i][1] for i in test))
        for mode,alpha in configurations:
            start=time.perf_counter()
            m=MultinomialNB(alpha=alpha)
            m.fit(data_sets[mode][train],labels[train])
            pred=m.predict(data_sets[mode][test])
            predictions[f"{mode}_a{alpha}"][test]=pred
            training_s[f"{mode}_a{alpha}"]+=time.perf_counter()-start
        fold_summaries.append({"fold":fold,"train":len(train),"test":len(test),
                               "unseen_truth":int((~seen_label[test]).sum())})
    def measure(ix,key):
        p=predictions[key][ix];truth=labels[ix]
        return {"n":len(ix),"correct":int((p==truth).sum()),
                "accuracy":round(float((p==truth).mean()),5) if len(ix) else None}
    summaries={k:measure(np.arange(len(rows)),k) for k in predictions}
    groups={}
    for dim,index in (("source",2),("encoding",3)):
        groups[dim]={}
        for label in sorted(set(r[index] for r in rows)):
            ix=np.array([i for i,r in enumerate(rows) if r[index]==label])
            groups[dim][label]={"n":len(ix),"models":{k:measure(ix,k) for k in predictions}}
    # One fixed latency batch includes extraction and complete model prediction.
    perf_index=np.arange(min(250,len(rows)))
    perf={}
    train=np.flatnonzero(folds!=0)
    for mode,alpha in configurations:
        m=MultinomialNB(alpha=alpha).fit(data_sets[mode][train],labels[train])
        duration=[]
        for _ in range(3):
            t=time.perf_counter_ns()
            for i in perf_index:
                u,p=features(rows[i][4])
                x=u if mode=="unigram" else p if mode=="ordered_transition" else np.r_[u,p]
                m.predict(x.reshape(1,-1))
            duration.append((time.perf_counter_ns()-t)/len(perf_index))
        perf[f"{mode}_a{alpha}"]={"median_ns_per_sample":round(sorted(duration)[1])}
    # Candidate viability evidence coverage, NO automatic pre-score masking.
    codeclist=("utf-8","shift-jis","euc-jp","big5","gb18030")
    evidence_counts=Counter()
    evidence_start=time.perf_counter()
    for r in rows:
        for label in codeclist:
            evidence_counts[f"{label}:{evidence(r[4],label)}"]+=1
    evidence_s=time.perf_counter()-evidence_start
    report={"scope":"independent architecture selection, no R21", "n":len(rows),
         "source":"chardet/test-data: development-only; competitor-owned fixtures",
         "exclusions":excluded,"folds":fold_summaries,
         "unseen_label_n":int((~seen_label).sum()),
         "candidates":summaries,"breakdowns":groups,
         "cost":{"feature_extraction_ms_each":round(1000*extraction_s/len(rows),4),
                 "logic_all_five_ms_each":round(1000*evidence_s/len(rows),4),
                 "fit_seconds":{k:round(v,3) for k,v in training_s.items()},
                 "runtime":perf,"rss_mib_linux":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2)},
         "logic_evidence_counts":dict(evidence_counts),
         "warning":"No R21 import or model download; 3-fold grouped development only, not independent release gate."}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"trace.jsonl").write_text("\n".join(json.dumps({"path":r[0],"source_group":r[1],"truth":r[3],
         "fold":int(folds[i]),"seen_in_training":bool(seen_label[i]),
         "predictions":{k:str(v[i]) for k,v in predictions.items()}}) for i,r in enumerate(rows))+"\n")
    print("NEW_ARCH_RESULT_START")
    print(json.dumps({k:v for k,v in report.items() if k not in ("breakdowns","logic_evidence_counts")},indent=2))
    print("NEW_ARCH_RESULT_END")

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--fixtures",required=True);a.add_argument("--output",default="research-results/new-arch-batch1")
    main(a.parse_args())
