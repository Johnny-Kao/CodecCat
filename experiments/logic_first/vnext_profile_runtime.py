"""Profile independent vNext detector phases without touching previous models."""
import argparse,codecs,hashlib,json,statistics,time,resource
from pathlib import Path
import numpy as np
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_bom import inspect_bom

def dataset(root):
    import sys
    sys.path.insert(0,str(root/"scripts"))
    from check_test_data import parse_dir_name
    out=[];seen=set()
    for p in sorted(root.glob("*/*")):
        if not p.is_file():continue
        e,_=parse_dir_name(p.parent.name)
        try:e=codecs.lookup(e).name
        except LookupError:continue
        b=p.read_bytes()[:4096]
        if not b:continue
        try:b.decode(e,"strict")
        except (UnicodeError,LookupError):continue
        h=hashlib.sha256(b).hexdigest()
        if h in seen:continue
        seen.add(h)
        out.append((b,e,p.stem))
    return out
def measure(fn):
    a=time.perf_counter_ns();fn();return time.perf_counter_ns()-a
def main(a):
    rows=dataset(Path(a.fixtures))
    train=rows[:2400];test=rows[2400:][:320]
    model=TrainableDetector.fit(train)
    timing={k:[] for k in ("bom","latin1","vectorizer","linear_score","full_inference","codec_top5")}
    for b,_,_ in test:
        s=inspect_bom(b)
        timing["bom"].append(measure(lambda:inspect_bom(b)))
        timing["latin1"].append(measure(lambda:b.decode("latin1")))
        t=b.decode("latin1")
        x=model.vectorizer.transform([t])
        timing["vectorizer"].append(measure(lambda:model.vectorizer.transform([t])))
        timing["linear_score"].append(measure(lambda:model.classifier.decision_function(x)))
        timing["full_inference"].append(measure(lambda:model.rank(b)))
        def codec():
            for c in model.classifier.classes_[:5]:
                try:b.decode(str(c),"strict")
                except (UnicodeError,LookupError):pass
        timing["codec_top5"].append(measure(codec))
    stats={}
    for k,seq in timing.items():
        q=sorted(seq)
        stats[k]={"p50_us":round(q[len(q)//2]/1000,2),"p95_us":round(q[int((len(q)-1)*.95)]/1000,2)}
    report={"n_train":len(train),"n_profile":len(test),"phases":stats,
            "rss_with_training_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
            "note":"same GitHub runner, per-operation warm timings; full_inference not sum of parts"}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("PROFILE_RESULT_START");print(json.dumps(report,indent=2));print("PROFILE_RESULT_END")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True)
    p.add_argument("--out",default="research-results/vnext-runtime-profile");main(p.parse_args())
