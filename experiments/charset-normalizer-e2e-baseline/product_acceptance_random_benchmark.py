"""One-shot R21 product acceptance: deterministic random unseen Common Crawl indexes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import random
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261014
HISTORICAL = {8, 12, 17, 21, 25, 30, 34, 39}
MODEL_SHA = "e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54"
MODELS = ("codeccat", "chardet7", "charset_normalizer")


def normalize(label):
    from h2_fused_candidate_mask_ab import normalize as norm
    return norm(label)


def select_crawls():
    import requests
    registry = requests.get("https://index.commoncrawl.org/collinfo.json", timeout=40)
    registry.raise_for_status()
    available = sorted({
        str(item.get("id", "")) for item in registry.json()
        if str(item.get("id", "")).startswith("CC-MAIN-2026-")
        and str(item.get("id", "")).rsplit("-", 1)[-1].isdigit()
        and int(str(item["id"]).rsplit("-", 1)[-1]) not in HISTORICAL
        and int(str(item["id"]).rsplit("-", 1)[-1]) < 40
    })
    if len(available) < 4:
        raise RuntimeError(f"Insufficient new published 2026 crawls: {available}")
    return sorted(random.Random(SEED).sample(available, 4))


def predict_one(model, data, runtime):
    if model == "codeccat":
        return runtime.rank(data)[0]
    if model == "chardet7":
        import chardet
        return normalize(chardet.detect(data).get("encoding"))
    from charset_normalizer import from_bytes
    best = from_bytes(data).best()
    return normalize(best.encoding) if best is not None else None


def worker(args):
    import gc
    with open(args.samples, "rb") as f:
        rows = pickle.load(f)
    sample_bytes = [r["data"] for r in rows]
    runtime = None
    if args.detector == "codeccat":
        from codeccat import load_bundle
        from codeccat.runtime import Runtime
        if hashlib.sha256(Path(args.model).read_bytes()).hexdigest() != MODEL_SHA:
            raise RuntimeError("Frozen model checksum mismatch")
        runtime = Runtime(load_bundle(Path(args.model)))
    # Exercise 4-route and BOM edge cases before benchmarking.
    synthetic = (b"hello", b"A"*512+b"\x80", b"\x80\x81"*300,
                 b"\x00\xff"*300, b"\xef\xbb\xbfhello")
    for v in synthetic:
        predict_one(args.detector, v, runtime)
    gc.collect()
    predictions = [predict_one(args.detector, x, runtime) for x in sample_bytes]
    # Warm full inference and avoid comparing initialization/import/IO time.
    for x in sample_bytes[:min(64, len(sample_bytes))]:
        predict_one(args.detector, x, runtime)
    timings = []
    for rep in range(7):
        start = time.perf_counter_ns()
        order = sample_bytes if rep % 2 == 0 else list(reversed(sample_bytes))
        for x in order:
            predict_one(args.detector, x, runtime)
        timings.append((time.perf_counter_ns()-start)/len(sample_bytes))
    report = {
        "detector": args.detector, "predictions": predictions,
        "median_ns_per_sample": statistics.median(timings),
        "min_ns_per_sample": min(timings), "max_ns_per_sample": max(timings),
        "samples_per_second": 1e9/statistics.median(timings),
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "repeats": 7
    }
    Path(args.result).write_text(json.dumps(report))
    print("WORKER_PASS", args.detector, len(predictions), flush=True)


def main(args):
    import requests
    from h2_fused_candidate_mask_ab import collect
    from codeccat.runtime import _signals, family
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(Path(args.model).read_bytes()).hexdigest() != MODEL_SHA:
        raise RuntimeError("R21 checksum mismatch")
    crawls = select_crawls()
    print("FROZEN_RANDOM_CRAWLS", json.dumps({"seed": SEED, "crawls": crawls}), flush=True)
    accepted, seen, counts = [], set(), {}
    for crawl in crawls:
        samples, paths, stats = collect(crawl)
        counts[crawl] = {"raw_accepted": len(samples), "paths": len(paths), "stats": stats}
        for s in samples:
            d = hashlib.sha256(s["data"]).hexdigest()
            if d in seen:
                continue
            seen.add(d)
            accepted.append({"data": s["data"], "sha256": d, "crawl": crawl,
                             "truth": normalize(s["label"]), "tier": s.get("tier"),
                             "route": _signals(s["data"]).route,
                             "family": family(normalize(s["label"]))})
        print("CRAWL_COLLECTED", crawl, counts[crawl]["raw_accepted"], flush=True)
    if len(accepted) < 200 or len({r["crawl"] for r in accepted}) < 4:
        raise RuntimeError("Insufficient or incomplete new-crawl coverage")
    dataset = output / "dataset.pkl"
    with dataset.open("wb") as f:
        pickle.dump(accepted, f, protocol=5)
    responses = {}
    for m in MODELS:
        path = output / (m + ".json")
        subprocess.run([sys.executable, __file__, "--worker", "--detector", m,
                        "--model", args.model, "--samples", str(dataset),
                        "--result", str(path)], check=True)
        responses[m] = json.loads(path.read_text())
        assert len(responses[m]["predictions"]) == len(accepted)
    def group_metrics(groups):
        result = []
        for key, subset in sorted(groups.items()):
            vals = {"key": key, "n": len(subset)}
            for m in MODELS:
                hits = sum(responses[m]["predictions"][i] == accepted[i]["truth"] for i in subset)
                vals[m] = {"correct": hits, "accuracy": hits / len(subset)}
            result.append(vals)
        return result
    groups = defaultdict(list)
    by_route, by_family = defaultdict(list), defaultdict(list)
    for i, r in enumerate(accepted):
        groups[r["crawl"]].append(i)
        by_route[r["route"]].append(i)
        by_family[r["family"]].append(i)
    all_idxs = list(range(len(accepted)))
    summary = {
        "status": "measured", "seed": SEED, "crawls": crawls,
        "limitations": ["Labels chosen by declared-header/body/BOM rules, not manually verified",
                        "Route-capped collection; not representative of global website traffic",
                        "New crawl index does not guarantee no overlap with original training corpus",
                        "Peak RSS is process total including loaded evaluation bytes",
                        "No policies trained or tuned on these samples"],
        "samples": len(accepted), "collection": counts,
        "overall": group_metrics({"all": all_idxs})[0],
        "by_crawl": group_metrics(groups),
        "by_route": group_metrics(by_route),
        "by_family": group_metrics(by_family),
        "performance": {m: {k: v for k,v in responses[m].items() if k != "predictions"} for m in MODELS}
    }
    baseline_ns = responses["codeccat"]["median_ns_per_sample"]
    summary["speedup_vs_codeccat"] = {
        m: responses[m]["median_ns_per_sample"]/baseline_ns for m in MODELS if m != "codeccat"
    }
    (output/"summary.json").write_text(json.dumps(summary, indent=2))
    (output/"sample-manifest.json").write_text(json.dumps([
        {**{k:v for k,v in r.items() if k != "data"},
         **{m: responses[m]["predictions"][i] for m in MODELS}}
        for i,r in enumerate(accepted)
    ], indent=2))
    dataset.unlink()  # Never publish raw third-party crawled bytes.
    for m in MODELS:
        (output/(m+".json")).unlink()
    print("PRODUCT_BENCHMARK_RESULT_START", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("PRODUCT_BENCHMARK_RESULT_END", flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--detector", choices=MODELS)
    ap.add_argument("--samples")
    ap.add_argument("--result")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="research-results/product-benchmark")
    args=ap.parse_args()
    worker(args) if args.worker else main(args)
