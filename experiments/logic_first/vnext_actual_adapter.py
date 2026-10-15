"""Evaluate actual vNext TrainableDetector (NOT standalone research scoring).

Competitor-owned corpus: development diagnostic, not independent release gate.
"""
import argparse,codecs,hashlib,json,resource,time,statistics
from pathlib import Path
from collections import Counter
from codeccat.vnext_strong_l2 import TrainableDetector

def corpus(root):
    import sys
    sys.path.insert(0,str(root/"scripts"))
    from check_test_data import parse_dir_name
    rows=[];seen=set()
    for p in sorted(root.glob("*/*")):
        if not p.is_file():continue
        enc,_=parse_dir_name(p.parent.name)
        try:label=codecs.lookup(enc).name
        except LookupError:continue
        b=p.read_bytes()
        if not b:continue
        try:b.decode(label,"strict")
        except (UnicodeError,LookupError):continue
        sha=hashlib.sha256(b).hexdigest()
        if sha in seen:continue
        seen.add(sha)
        group=p.stem.lower().replace("culturax_mc4_","culturax_").replace("culturax_oscar-2301_","culturax_")
        fold=int(hashlib.sha256(group.encode()).hexdigest()[:12],16)%3
        rows.append((b[:4096],label,group,fold,sha))
    return rows

def run(args):
    rows=corpus(Path(args.fixtures))
    fold_reports=[];traces=[];latencies=[]
    for fold in range(3):
        training=[(b,label,group) for b,label,group,k,_ in rows if k!=fold]
        evaluation=[(b,label,group,sha) for b,label,group,k,sha in rows if k==fold]
        assert {g for _,_,g in training}.isdisjoint({g for _,_,g,_ in evaluation})
        model=TrainableDetector.fit(training)
        counts=Counter()
        for raw,truth,group,sha in evaluation:
            start=time.perf_counter_ns()
            answer=model.rank(raw,final=(len(raw)<4096))
            latencies.append(time.perf_counter_ns()-start)
            counts["n"]+=1
            counts["correct"]+=int(answer.encoding==truth)
            counts["unknown"]+=int(answer.encoding is None)
            counts["truth_retained"]+=int(truth in answer.candidates)
            counts["truth_unseen"]+=int(truth not in set(model.classifier.classes_))
            counts["bom_fast"]+=int(answer.bom_hint is not None and len(answer.candidates)==1)
            traces.append({"sha":sha,"source_group":group,"fold":fold,"truth":truth,
                           "actual":answer.encoding,"truth_retained":truth in answer.candidates})
        fold_reports.append({"fold":fold,**dict(counts)})
        print("PRODUCTION_ADAPTER_FOLD_DONE",fold,flush=True)
    n=sum(z["n"] for z in fold_reports)
    hits=sum(z["correct"] for z in fold_reports)
    sorted_ns=sorted(latencies)
    report={"n":n,"correct":hits,"accuracy":round(hits/n,5),"folds":fold_reports,
      "latency_ns_p50":sorted_ns[len(sorted_ns)//2],
      "latency_ns_p95":sorted_ns[int((len(sorted_ns)-1)*.95)],
      "process_peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
      "source":"chardet/test-data: competitor-owned development diagnostic",
      "note":"actual vnext_strong_l2.TrainableDetector, grouped three-fold; maxrss includes training; no R21"}
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/"report.json").write_text(json.dumps(report,indent=2))
    (dest/"trace.jsonl").write_text("\n".join(map(json.dumps,traces))+"\n")
    print("ACTUAL_ADAPTER_RESULT_START");print(json.dumps(report,indent=2));print("ACTUAL_ADAPTER_RESULT_END")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True)
    p.add_argument("--out",default="research-results/vnext-actual-adapter")
    run(p.parse_args())
