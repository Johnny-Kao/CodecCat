"""Paired L1->L2 integration: safety, unchanged prediction and inference cost.

The corpus is previously inspected. This is a compatibility/regression gate,
NOT a fresh accuracy holdout or evidence of competitor superiority.
"""
from __future__ import annotations
import argparse,codecs,collections,hashlib,json,random,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_multidist import family,gather,norm
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector

def percentile(numbers,fraction):
    xs=sorted(numbers)
    return xs[round((len(xs)-1)*fraction)]

def external_rows(a,known_bytes,known_text):
    rng=random.Random(20261015)
    strata=collections.defaultdict(list)
    seen=set()
    for source,root in (("uchardet",Path(a.uchardet)),("unicode-suite",Path(a.unicode_suite))):
        for filename,label in gather(root,source):
            raw=filename.read_bytes()[:4096]
            if not raw:continue
            try:decoded=codecs.getincrementaldecoder(label)("strict").decode(raw,final=len(raw)<4096)
            except (UnicodeError,LookupError):continue
            sha=hashlib.sha256(raw).hexdigest()
            text_sha=hashlib.sha256(decoded.encode()).hexdigest()
            if sha in known_bytes or text_sha in known_text or sha in seen:continue
            seen.add(sha)
            strata[(source,family(label))].append((raw,label,source))
    selected=[]
    for stratum in sorted(strata):
        rows=strata[stratum]
        rng.shuffle(rows)
        selected+=rows[:12]
    return selected,{"/".join(key):len(rows) for key,rows in sorted(strata.items())}

def run(a):
    original=[(b,y,g) for b,y,g,fold,sha in corpus(Path(a.train))]
    training,_=residual_subset(original)
    known_bytes={hashlib.sha256(b).hexdigest() for b,y,g in original}
    known_text=set()
    for b,y,g in original:
        try:known_text.add(hashlib.sha256(b.decode(y).encode()).hexdigest())
        except UnicodeError:pass
    rows,strata=external_rows(a,known_bytes,known_text)
    if len(rows)<30:raise RuntimeError("too few nonoverlap source rows")
    result={"source":"pre-inspected uchardet and synthetic UnicodeTestSuite",
        "purpose":"identical frozen-L2 predictions and real structure-first cost, not new accuracy confirmation",
        "rows":len(rows),"strata":strata,"profiles":{}}
    for profile in ("stable23","experimental13"):
        model=TrainableDetector.fit(training,feature_profile=profile)
        old_ns=[];new_ns=[];disagreements=[];truth_eliminated=0
        for index,(raw,truth,source) in enumerate(rows):
            prefix=raw
            # Compare both paths with identical prefix/final semantics.
            evidence=model.analyze_structure(prefix,final=len(prefix)<4096)
            truth_eliminated+=int(truth in evidence.impossible)
            if index%2:
                start=time.perf_counter_ns()
                after=model.rank(prefix,final=len(prefix)<4096,structural_first=True)
                new_ns.append(time.perf_counter_ns()-start)
                start=time.perf_counter_ns()
                before=model.rank(prefix,final=len(prefix)<4096)
                old_ns.append(time.perf_counter_ns()-start)
            else:
                start=time.perf_counter_ns()
                before=model.rank(prefix,final=len(prefix)<4096)
                old_ns.append(time.perf_counter_ns()-start)
                start=time.perf_counter_ns()
                after=model.rank(prefix,final=len(prefix)<4096,structural_first=True)
                new_ns.append(time.perf_counter_ns()-start)
            if before.encoding!=after.encoding or before.candidates!=after.candidates:
                disagreements.append({"sha":hashlib.sha256(prefix).hexdigest(),
                    "truth":truth,"source":source,"old":before.encoding,"new":after.encoding,
                    "old_candidates":before.candidates,"new_candidates":after.candidates})
        summary={"n":len(rows),"truth_eliminated":truth_eliminated,
            "changed_predictions":len(disagreements),
            "first_disagreements":disagreements[:10],
            "baseline_ns_p50":percentile(old_ns,.50),
            "baseline_ns_p95":percentile(old_ns,.95),
            "structure_first_ns_p50":percentile(new_ns,.50),
            "structure_first_ns_p95":percentile(new_ns,.95),
            "p50_latency_ratio":round(percentile(new_ns,.50)/max(1,percentile(old_ns,.50)),3),
            "p95_latency_ratio":round(percentile(new_ns,.95)/max(1,percentile(old_ns,.95)),3)}
        result["profiles"][profile]=summary
        print("L1_L2_PROFILE_DONE",profile,summary["changed_predictions"],flush=True)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    (output/"report.json").write_text(json.dumps(result,indent=2))
    print("L1_L2_INTEGRATION_BEGIN");print(json.dumps(result,indent=2));print("L1_L2_INTEGRATION_END")
    if any(v["truth_eliminated"] or v["changed_predictions"] for v in result["profiles"].values()):
        raise AssertionError("L1->L2 gating changed a prediction or incorrectly eliminated truth")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True)
    p.add_argument("--uchardet",required=True)
    p.add_argument("--unicode-suite",required=True)
    p.add_argument("--output",default="research-results/vnext-l1-l2-integration")
    run(p.parse_args())
