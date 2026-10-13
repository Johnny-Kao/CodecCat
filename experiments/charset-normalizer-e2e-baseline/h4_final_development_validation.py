from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from codeccat import load_bundle
from codeccat.runtime import Runtime

from h2_fused_candidate_mask_ab import collect, dedupe, normalize
from h3_comprehensive_full_b_necessity import baseline_and_full, rank_policy

SELECTION_CRAWLS = ("CC-MAIN-2026-39", "CC-MAIN-2026-34")
VALIDATION_CRAWLS = ("CC-MAIN-2026-21", "CC-MAIN-2026-17")
POLICY = "same_family_0p75"


def fingerprint(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def evaluate(rows, runtime):
    control_hits = candidate_hits = 0
    control_triggers = candidate_triggers = 0
    skipped = skipped_beneficial = skipped_noop = 0

    for s in rows:
        truth = normalize(s["label"])
        info = baseline_and_full(runtime, s["data"])
        control_rank = rank_policy(runtime, s["data"], "control")
        candidate_rank = rank_policy(runtime, s["data"], POLICY)

        control_hits += int(bool(control_rank) and control_rank[0] == truth)
        candidate_hits += int(bool(candidate_rank) and candidate_rank[0] == truth)
        control_triggers += int(info["current_escalate"])

        # mirror frozen policy logic without introducing any new condition
        from h3_comprehensive_full_b_necessity import should_escalate
        cand_escalate = should_escalate(info, POLICY)
        candidate_triggers += int(cand_escalate)

        if info["current_escalate"] and not cand_escalate:
            skipped += 1
            b = info["baseline_final"][0] if info["baseline_final"] else None
            f = info["full_final"][0] if info["full_final"] else None
            bc = b == truth
            fc = f == truth
            skipped_beneficial += int((not bc) and fc)
            skipped_noop += int(b == f)

    n=len(rows)
    return {
        "n":n,
        "control_hits":control_hits,
        "candidate_hits":candidate_hits,
        "control_accuracy":control_hits/max(1,n),
        "candidate_accuracy":candidate_hits/max(1,n),
        "delta_hits":candidate_hits-control_hits,
        "control_triggers":control_triggers,
        "candidate_triggers":candidate_triggers,
        "trigger_reduction_n":control_triggers-candidate_triggers,
        "trigger_reduction_fraction":(control_triggers-candidate_triggers)/max(1,control_triggers),
        "skipped_full_b":skipped,
        "skipped_beneficial_full_b":skipped_beneficial,
        "skipped_noop_full_b":skipped_noop,
    }


def benchmark(rows, runtime, repeats=21):
    data=[s["data"] for s in rows]
    def bench(policy):
        vals=[]
        sink=0
        for _ in range(repeats):
            t0=time.perf_counter_ns()
            for d in data:
                sink += len(rank_policy(runtime,d,policy))
            vals.append((time.perf_counter_ns()-t0)/max(1,len(data)))
        return {
            "median_ns_per_sample":statistics.median(vals),
            "min_ns_per_sample":min(vals),
            "max_ns_per_sample":max(vals),
            "sink":sink,
        }
    a=bench("control")
    b=bench(POLICY)
    return {
        "repeats":repeats,
        "control":a,
        "candidate":b,
        "delta_ns":b["median_ns_per_sample"]-a["median_ns_per_sample"],
        "delta_fraction":b["median_ns_per_sample"]/a["median_ns_per_sample"]-1.0,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--expected-model-sha256",required=True)
    args=ap.parse_args()

    p=Path(args.model)
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != args.expected_model_sha256:
        raise RuntimeError(f"model SHA mismatch: {actual}")
    runtime=Runtime(load_bundle(p))

    selection_fps=set()
    selection_meta={}
    for crawl in SELECTION_CRAWLS:
        rows, paths, stats=collect(crawl)
        rows=dedupe(rows)
        selection_meta[crawl]={"n":len(rows),"warc_paths_considered":len(paths),"stats":stats}
        selection_fps.update(fingerprint(s["data"]) for s in rows)

    all_validation=[]
    per={}
    collection={}
    overlap_removed={}
    for crawl in VALIDATION_CRAWLS:
        rows, paths, stats=collect(crawl)
        rows=dedupe(rows)
        before=len(rows)
        rows=[s for s in rows if fingerprint(s["data"]) not in selection_fps]
        overlap_removed[crawl]=before-len(rows)
        all_validation.extend(rows)
        per[crawl]=evaluate(rows,runtime)
        collection[crawl]={"n_after_overlap_filter":len(rows),"n_before_overlap_filter":before,"warc_paths_considered":len(paths),"stats":stats}

    pooled=dedupe(all_validation)
    pooled_eval=evaluate(pooled,runtime)
    timing=benchmark(pooled,runtime)

    print(json.dumps({
        "phase":"h4_final_development_validation",
        "policy":{
            "candidate":POLICY,
            "candidate_frozen_before_validation":True,
            "parameters_tuned":False,
            "retrained":False,
            "r22_used":False,
            "release_candidate_modified":False,
            "selection_crawls":SELECTION_CRAWLS,
            "validation_crawls":VALIDATION_CRAWLS,
            "exact_byte_overlap_with_selection_removed":True,
        },
        "selection_meta":selection_meta,
        "validation_collection":collection,
        "overlap_removed":overlap_removed,
        "per_crawl":per,
        "pooled":pooled_eval,
        "timing":timing,
        "gate":{
            "each_crawl_nonnegative":all(v["delta_hits"]>=0 for v in per.values()),
            "pooled_nonnegative":pooled_eval["delta_hits"]>=0,
            "zero_skipped_beneficial":pooled_eval["skipped_beneficial_full_b"]==0,
            "trigger_reduced":pooled_eval["trigger_reduction_n"]>0,
            "runtime_improved":timing["delta_ns"]<0,
        },
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
