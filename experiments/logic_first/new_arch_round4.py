"""Round 4: true L1 / trainable L2 / selective L3 paired ablation.
Independent from R21. Competitor-owned labelled corpus: development-only.
"""
import argparse,codecs,hashlib,json,time,resource
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

def canonical(s):
    try:return codecs.lookup(s).name.replace("_","-")
    except LookupError:return None
def load(root):
    import sys
    sys.path.insert(0,str(root/"scripts"))
    from check_test_data import parse_dir_name
    rows=[];seen=set()
    for p in sorted(root.glob("*/*")):
        if not p.is_file():continue
        enc,lang=parse_dir_name(p.parent.name);label=canonical(enc)
        if not label:continue
        raw=p.read_bytes()
        if not raw:continue
        try:raw.decode(enc,"strict")
        except (UnicodeDecodeError,LookupError):continue
        digest=hashlib.sha256(raw).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        group=p.stem.lower().replace("culturax_mC4_","culturax_").replace("culturax_OSCAR-2301_","culturax_")
        fold=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%3
        rows.append((label,raw[:4096],fold,group,p.name.startswith("culturax_")))
    return rows
def viable(data,label):
    # Incomplete terminal code units treated as UNKNOWN, never hard rejected.
    for trim in range(4):
        if trim and len(data)<=trim:break
        try:(data[:-trim] if trim else data).decode(label,"strict");return True
        except (UnicodeDecodeError,LookupError):pass
    return False
def l1(data,labels):
    # Strong BOM evidence is a priority only for compatible decoder labels.
    bom=(data.startswith(b"\xef\xbb\xbf"),data.startswith(b"\xff\xfe"),data.startswith(b"\xfe\xff"))
    if any(bom):
        targets=("utf-8-sig", "utf-16", "utf-16")
        c=targets[bom.index(True)]
        if c in labels and viable(data,c):return [c],True
    # Avoid unsafe family-wide rejection from arbitrary middle/truncated prefix.
    survivors=[c for c in labels if viable(data,c)]
    return survivors if survivors else list(labels),False
def choose(top,data,need_l3):
    if not need_l3:return top[0]
    # Recheck full available bytes for only top three. Keep stable model
    # ordering; only provably impossible candidates can be excluded.
    for c in top[:3]:
        if viable(data,c):return c
    return top[0]
def run(a):
    rows=load(Path(a.fixtures));assert len(rows)>1500
    labels=np.array([r[0] for r in rows]);folds=np.array([r[2] for r in rows])
    result={str(z):{} for z in (128,1024,4096)}
    traces=[];start=time.perf_counter()
    for size in (128,1024,4096):
        samples=[r[1][:size] for r in rows]
        vec=HashingVectorizer(analyzer="char",ngram_range=(1,2),n_features=4096,
            alternate_sign=False,norm="l1",lowercase=False,dtype=np.float32)
        X=vec.transform([x.decode("latin1") for x in samples])
        outcome={k:Counter() for k in ("l2_only","l1_l2","l2_l3","full")}
        layer=Counter();latency={k:[] for k in outcome}
        for f in (0,1,2):
            train=np.flatnonzero(folds!=f);test=np.flatnonzero(folds==f)
            assert {rows[i][3] for i in train}.isdisjoint({rows[i][3] for i in test})
            model=MultinomialNB(alpha=0.1).fit(X[train],labels[train])
            classes=list(model.classes_)
            logp=model.predict_log_proba(X[test])
            for j,i in enumerate(test):
                raw=samples[i];truth=labels[i];order=np.argsort(logp[j])[::-1]
                rank=[classes[k] for k in order]
                tic=time.perf_counter_ns()
                survivors,shortcut=l1(raw,classes)
                l1_ns=time.perf_counter_ns()-tic
                survives=truth in survivors
                layer["l1_false_exclusions"]+=int(not survives and truth in classes)
                layer["l1_survivors_total"]+=len(survivors)
                layer["l1_shortcut"]+=int(shortcut)
                base=rank[0]
                filtered=next((c for c in rank if c in survivors),base)
                margin=float(logp[j,order[0]]-logp[j,order[1]]) if len(order)>1 else 0.0
                need_l3=margin<1.0
                layer["l3_invocations"]+=int(need_l3)
                # Separate control vs selective post-score verifier
                post=choose(rank,raw,need_l3)
                full=filtered if shortcut else choose([c for c in rank if c in survivors] or rank,raw,need_l3)
                pred={"l2_only":base,"l1_l2":filtered,"l2_l3":post,"full":full}
                for k,v in pred.items():
                    outcome[k]["correct"]+=int(v==truth)
                    outcome[k]["n"]+=1
                layer["l1_help"]+=int(filtered==truth and base!=truth)
                layer["l1_harm"]+=int(filtered!=truth and base==truth)
                layer["l3_help"]+=int(full==truth and filtered!=truth)
                layer["l3_harm"]+=int(full!=truth and filtered==truth)
                latency["full"].append(l1_ns)
                if size==4096:
                    traces.append({"sha":hashlib.sha256(raw).hexdigest(),"fold":f,
                      "truth":truth,"predictions":pred,"survives_l1":survives,
                      "survivor_count":len(survivors),"l3_called":need_l3})
        result[str(size)]={"models":{k:{"correct":v["correct"],"n":v["n"],
                            "accuracy":round(v["correct"]/max(1,v["n"]),5)} for k,v in outcome.items()},
                   "layer":dict(layer),"l1_median_ns":int(np.median(latency["full"]))}
        print("ROUND4_SIZE_DONE",size,flush=True)
    elapsed=time.perf_counter()-start
    report={"round":4,"corpus":"chardet/test-data; competitor-owned development-only","n":len(rows),
            "design":"stable structural L1 / trained byte L2 / lazy top3 strict L3",
            "results":result,"wall_seconds":round(elapsed,3),
            "mixed_process_peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
            "limitations":["L1 checks candidate decoding eagerly; this is diagnostic correctness, NOT optimized fast-path",
             "L3 currently strict feasibility only and may offer no improvements beyond L1",
             "Latency is L1-only, not end-to-end; full performance signoff is NOT met",
             "Training corpus owned by competing detector; not independent validation"]}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"trace.jsonl").write_text("\n".join(json.dumps(t) for t in traces)+"\n")
    print("ROUND4_RESULT_START");print(json.dumps(report,indent=2));print("ROUND4_RESULT_END")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True)
    p.add_argument("--output",default="research-results/new-arch-round4");run(p.parse_args())
