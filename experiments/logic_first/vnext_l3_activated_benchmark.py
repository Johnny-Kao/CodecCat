"""Benchmark the ACTUALLY INVOKED L3 path, not only its cold/skip path.

All model fitting uses fold != 0, and candidate probes use held-out fold 0.
No source-independent accuracy claims from this development performance run.
"""
from __future__ import annotations
import argparse, collections, json, platform, resource, statistics, sys, time
from pathlib import Path

from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_l3_language import LanguageCoherence
from codeccat.vnext_decoding import decode
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset


def percentile(nums, fraction):
    if not nums:
        return None
    return round(sorted(nums)[round((len(nums)-1)*fraction)]/1e6, 5)


def run(a):
    rows=corpus(Path(a.training))
    training=[(b,y,g) for b,y,g,k,_ in rows if k != 0]
    dev=[(b,y,g,h) for b,y,g,k,h in rows if k == 0]
    assert {g for _,_,g in training}.isdisjoint({g for _,_,g,_ in dev})
    focused,_=residual_subset(training)
    model=TrainableDetector.fit(focused)
    lm=LanguageCoherence.fit(training)
    triggered=[]
    eligible_total=0
    for data,label,group,sha in dev:
        r=model.rank(data,final=len(data)<4096)
        candidate=lm.rerank(r,data,final=len(data)<4096)
        if candidate.called:
            eligible_total += 1
            if len(triggered) < a.cases:
                triggered.append((data,label,group,sha,candidate.changed))
    if len(triggered)<min(a.cases,8):
        raise RuntimeError(f"too few L3-activated samples: {len(triggered)}")
    rows_out=[]
    total_baseline,total_l3=[],[]
    for raw,truth,group,sha,changed in triggered:
        for _ in range(5):
            decode(model, raw)
            actual=decode(model,raw,language_model=lm)
            if not actual.l3_called:
                raise AssertionError("latency probe did not exercise L3")
        b_ns,l_ns,b_cpu,l_cpu=[],[],[],[]
        for _ in range(a.iterations):
            start=time.perf_counter_ns()
            cpu=time.process_time_ns()
            baseline=decode(model,raw)
            b_cpu.append(time.process_time_ns()-cpu)
            b_ns.append(time.perf_counter_ns()-start)
            start=time.perf_counter_ns()
            cpu=time.process_time_ns()
            enabled=decode(model,raw,language_model=lm)
            l_cpu.append(time.process_time_ns()-cpu)
            l_ns.append(time.perf_counter_ns()-start)
        assert enabled.l3_called
        assert baseline.text is not None or not baseline.accepted
        total_baseline.extend(b_ns);total_l3.extend(l_ns)
        rows_out.append({
            "sha":sha, "group":group, "label":truth, "bytes":len(raw),
            "called":enabled.l3_called, "changed":changed,
            "baseline_p50_ms":percentile(b_ns,.5),
            "baseline_p95_ms":percentile(b_ns,.95),
            "l3_p50_ms":percentile(l_ns,.5),
            "l3_p95_ms":percentile(l_ns,.95),
            "l3_cpu_p50_ms":percentile(l_cpu,.5),
            "baseline_cpu_p50_ms":percentile(b_cpu,.5),
            "added_p50_ms":round((statistics.median(l_ns)-statistics.median(b_ns))/1e6,5),
        })
    report={
        "schema":"vnext-l3-activated-resource-v1",
        "source":"source-disjoint chardet/test-data fold 0 performance development",
        "arch":platform.machine(), "python":platform.python_version(),
        "candidate_trigger_count_in_fold":eligible_total,
        "n_distinct_activated_samples":len(rows_out),
        "iterations_each":a.iterations,
        "baseline_p50_ms":percentile(total_baseline,.5),
        "baseline_p95_ms":percentile(total_baseline,.95),
        "l3_p50_ms":percentile(total_l3,.5),
        "l3_p95_ms":percentile(total_l3,.95),
        "peak_process_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "caveat":"Peak RSS includes training, not incremental inference-only RSS; run alongside existing separate-process model load figures. Sequential paired timings are descriptive, not a fixed SLO.",
        "cases":rows_out,
    }
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("ACTIVATED_L3_PERFORMANCE_BEGIN",flush=True)
    print(json.dumps(report,indent=2,sort_keys=True),flush=True)
    print("ACTIVATED_L3_PERFORMANCE_END",flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--training", required=True)
    p.add_argument("--cases",type=int,default=12)
    p.add_argument("--iterations",type=int,default=60)
    p.add_argument("--output",default="research-results/vnext-l3-activated")
    run(p.parse_args())
