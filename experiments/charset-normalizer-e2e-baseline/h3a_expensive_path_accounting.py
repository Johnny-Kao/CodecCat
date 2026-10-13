from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from codeccat import load_bundle
from codeccat.features import baseline_from_array, full_b_extra_from_array, hmt768_array
from codeccat.runtime import Runtime, _context, _signals

from h2_fused_candidate_mask_ab import collect, dedupe, normalize


def classify_one(bundle, data):
    signals = _signals(data)
    array = hmt768_array(data)
    baseline_vector = baseline_from_array(array)
    bmodel = bundle.baseline.route_models[signals.route]
    bctx = _context(data, signals, bmodel, baseline_vector)
    threshold = bundle.gate_thresholds.get(signals.route)
    margin = (
        float(bctx.sorted_scores[0] - bctx.sorted_scores[1])
        if len(bctx.sorted_scores) >= 2
        else float("inf")
    )
    full_b = (
        bundle.full_b is not None
        and threshold is not None
        and signals.route in ("U", "RH")
        and margin <= threshold
    )
    rl_spec = signals.route == "RL" and bundle.rl_specialist is not None
    return signals, array, baseline_vector, bctx, full_b, rl_spec


def evaluate(rows, bundle):
    out = Counter()
    routes = defaultdict(Counter)
    margins = defaultdict(list)
    for s in rows:
        signals, _, _, bctx, full_b, rl_spec = classify_one(bundle, s["data"])
        out["n"] += 1
        out["full_b"] += int(full_b)
        out["rl_specialist_path"] += int(rl_spec)
        out["baseline_only"] += int(not full_b and not rl_spec)
        routes[signals.route]["n"] += 1
        routes[signals.route]["full_b"] += int(full_b)
        routes[signals.route]["rl_specialist_path"] += int(rl_spec)
        if len(bctx.sorted_scores) >= 2:
            margins[signals.route].append(float(bctx.sorted_scores[0] - bctx.sorted_scores[1]))
    return {
        "n": out["n"],
        "full_b_n": out["full_b"],
        "full_b_rate": out["full_b"] / max(1, out["n"]),
        "rl_specialist_path_n": out["rl_specialist_path"],
        "rl_specialist_path_rate": out["rl_specialist_path"] / max(1, out["n"]),
        "baseline_only_n": out["baseline_only"],
        "routes": {
            r: {
                "n": c["n"],
                "full_b_n": c["full_b"],
                "full_b_rate": c["full_b"] / max(1, c["n"]),
                "rl_specialist_path_n": c["rl_specialist_path"],
            }
            for r,c in routes.items()
        },
        "margin_summary": {
            r: {
                "n": len(v),
                "p10": float(np.quantile(v, .10)) if v else None,
                "p25": float(np.quantile(v, .25)) if v else None,
                "p50": float(np.quantile(v, .50)) if v else None,
                "p75": float(np.quantile(v, .75)) if v else None,
                "p90": float(np.quantile(v, .90)) if v else None,
            }
            for r,v in margins.items()
        },
    }


def bench(rows, bundle, repeats=21):
    runtime = Runtime(bundle)
    prepared = []
    for s in rows:
        signals, array, baseline_vector, bctx, full_b, rl_spec = classify_one(bundle, s["data"])
        prepared.append((s["data"], signals, array, baseline_vector, bctx, full_b, rl_spec))

    def measure(fn, subset):
        vals = []
        sink = 0
        if not subset:
            return {"n":0,"median_ns_per_sample":None,"min_ns_per_sample":None,"max_ns_per_sample":None,"sink":0}
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            for item in subset:
                sink += int(fn(item))
            vals.append((time.perf_counter_ns() - t0) / len(subset))
        return {
            "n": len(subset),
            "median_ns_per_sample": statistics.median(vals),
            "min_ns_per_sample": min(vals),
            "max_ns_per_sample": max(vals),
            "sink": sink,
        }

    all_items = prepared
    full_items = [x for x in prepared if x[5]]
    nonfull_items = [x for x in prepared if not x[5]]
    rl_items = [x for x in prepared if x[6]]

    def full_runtime(item):
        return len(runtime.rank(item[0]))

    def baseline_feature_stage(item):
        data=item[0]
        sig=_signals(data)
        arr=hmt768_array(data)
        vec=baseline_from_array(arr)
        model=bundle.baseline.route_models[sig.route]
        ctx=_context(data,sig,model,vec)
        return len(ctx.rank)

    def full_extra_stage(item):
        data, sig, arr, basevec, *_ = item
        extra=full_b_extra_from_array(arr, sig.route)
        fvec=np.concatenate([basevec,extra])
        model=bundle.full_b.route_models[sig.route]
        ctx=_context(data,sig,model,fvec)
        return len(ctx.rank)

    return {
        "repeats": repeats,
        "runtime_all": measure(full_runtime, all_items),
        "runtime_full_b_triggered": measure(full_runtime, full_items),
        "runtime_non_full_b": measure(full_runtime, nonfull_items),
        "baseline_feature_context": measure(baseline_feature_stage, all_items),
        "full_b_extra_plus_context": measure(full_extra_stage, full_items),
        "runtime_rl_path": measure(full_runtime, rl_items),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--expected-model-sha256", required=True)
    args=ap.parse_args()

    path=Path(args.model)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != args.expected_model_sha256:
        raise RuntimeError(f"model SHA mismatch: {actual}")
    bundle=load_bundle(path)

    all_rows=[]
    per={}
    collection={}
    for crawl in ("CC-MAIN-2026-39","CC-MAIN-2026-34"):
        rows, paths, stats=collect(crawl)
        rows=dedupe(rows)
        all_rows.extend(rows)
        per[crawl]=evaluate(rows,bundle)
        collection[crawl]={"n":len(rows),"warc_paths_considered":len(paths),"stats":stats}

    pooled=dedupe(all_rows)
    report={
        "phase":"h3a_expensive_path_accounting",
        "policy":{
            "r22_used":False,
            "parameters_tuned":False,
            "retrained":False,
            "release_candidate_modified":False,
            "purpose":"identify expensive runtime paths before proposing any new mechanism",
        },
        "collection":collection,
        "per_crawl":per,
        "pooled":evaluate(pooled,bundle),
        "timing":bench(pooled,bundle),
    }
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
