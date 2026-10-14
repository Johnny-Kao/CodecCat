"""Independent-process memory baseline for the three frozen product detectors.

Measures cold import+initialization, fixed-input steady RSS, tracemalloc deltas,
per-call peak RSS (Linux), and 4KiB/64KiB wall-clock inference. No Common Crawl download.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

SHA="e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54"
DETECTORS=("codeccat","chardet7","charset_normalizer")
SIZE=(4096,65536)

def rss_kib():
    # Linux instantaneous RSS; ru_maxrss alone cannot be used for per-object costs.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("Linux procfs required")

def process_one(detector, model, out):
    empty=rss_kib()
    tracemalloc.start()
    if detector=="codeccat":
        import hashlib
        from codeccat import load_bundle
        from codeccat.runtime import Runtime
        assert hashlib.sha256(Path(model).read_bytes()).hexdigest()==SHA
        rt=Runtime(load_bundle(Path(model)))
        predict=rt.rank
    elif detector=="chardet7":
        import chardet
        predict=lambda data: chardet.detect(data).get("encoding")
    else:
        from charset_normalizer import from_bytes
        predict=lambda data: (from_bytes(data).best())
    after_init=rss_kib()
    results=[]
    for size in SIZE:
        # Deterministic mixed bytes, valid and invalid UTF-8; no benchmark labels.
        data=bytes((i*131+17)%256 for i in range(size))
        predict(data)
        gc.collect()
        warm=rss_kib()
        times=[]
        for _ in range(20):
            t=time.perf_counter_ns()
            predict(data)
            times.append(time.perf_counter_ns()-t)
        after=rss_kib()
        results.append({"size_bytes":size,"warm_rss_kib":warm,"post_rss_kib":after,
                        "median_ns":sorted(times)[len(times)//2]})
    current, peak=tracemalloc.get_traced_memory()
    output={"detector":detector,"rss_before_import_kib":empty,
            "rss_after_init_kib":after_init,"rss_init_increment_kib":after_init-empty,
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "tracemalloc_current_bytes":current,"tracemalloc_peak_bytes":peak,
            "results":results}
    Path(out).write_text(json.dumps(output,indent=2))
    print("MEMORY_WORKER_PASS", detector, flush=True)

def main(args):
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    reports={}
    for det in DETECTORS:
        dest=out/f"{det}.json"
        subprocess.run([sys.executable,__file__,"--worker","--detector",det,
                        "--model",args.model,"--out",str(dest)],check=True)
        reports[det]=json.loads(dest.read_text())
    (out/"memory-baseline.json").write_text(json.dumps({
        "contract":"Separate Python processes; Linux Ubuntu runner; fixed synthetic bytes",
        "caveats":["Import baseline and detector allocation include interpreter/module caches",
                   "RSS is OS-process RSS and is not exclusive allocations of detector",
                   "Synthetic bytes have no accuracy ground truth",
                   "RSS readings at fixed checkpoints may miss transient peaks"],
        "detectors":reports},indent=2))
    print("MEMORY_BASELINE_RESULT_START", flush=True)
    print(json.dumps(reports,indent=2),flush=True)
    print("MEMORY_BASELINE_RESULT_END",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--detector",choices=DETECTORS)
    parser.add_argument("--model",required=True)
    parser.add_argument("--out",default="research-results/memory-baseline")
    args=parser.parse_args()
    process_one(args.detector,args.model,args.out) if args.worker else main(args)
