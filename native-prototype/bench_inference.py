"""Comparable Alpha vs Native API performance on identical fixed input bytes.

Run in two fresh interpreters with mutually exclusive codeccat installs:
  PYTHONPATH=src python native-prototype/bench_inference.py --variant alpha
  .native-clean/bin/python native-prototype/bench_inference.py --variant native

Published source Alpha wheels use NumPy and sklearn. Native wheel does not.
Execution is CPU-bound per-process in one CI runner. Benchmark does not
claim universal speedup or capture platform-neutral RSS.
"""
from __future__ import annotations
import argparse,json,os,resource,statistics,time
from pathlib import Path

def percentile(values,p):
    seq=sorted(values)
    return seq[min(len(seq)-1,int((len(seq)-1)*p))]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("alpha","native"),required=True)
    p.add_argument("--model",default="src/codeccat/models/stable23.zip")
    args=p.parse_args()
    begin=time.perf_counter()
    from codeccat import CodecCat
    import_ready=time.perf_counter()
    if args.variant=="alpha":
        detector=CodecCat.from_files(args.model)
    else:
        detector=CodecCat()
    loaded=time.perf_counter()
    samples={
        "ascii_5":b"hello",
        "latin_80":("caf\u00e9 cr\u00e8me! "*6).encode("cp1252"),
        "utf8_512":("日本語とcafé。"*45).encode("utf-8")[:512],
        "utf8_4096":("日本語とcafé。"*400).encode("utf-8")[:4096],
        "cp1252_4096":("café crème façade. "*220).encode("cp1252")[:4096],
        "large_100k":("日本語とcafé。"*10000).encode("utf-8")[:100000],
    }
    report={}
    for label,data in samples.items():
        # One warmup + 120 iterations; record per-call latency and CPU.
        detector.detect(data)
        elapsed=[];cpu=[]
        for _ in range(120):
            a=time.perf_counter_ns()
            c=time.process_time_ns()
            detector.detect(data)
            cpu.append((time.process_time_ns()-c)/1e6)
            elapsed.append((time.perf_counter_ns()-a)/1e6)
        report[label]={"bytes":len(data),
            "p50_ms":round(statistics.median(elapsed),4),
            "p95_ms":round(percentile(elapsed,.95),4),
            "cpu_p50_ms":round(statistics.median(cpu),4)}
    # Linux ru_maxrss is measured in KiB.
    report["process"]={
        "import_time_ms":round((import_ready-begin)*1000,3),
        "model_load_ms":round((loaded-import_ready)*1000,3),
        "maxrss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,3),
    }
    print("NATIVE_PERF_RESULT",json.dumps({"variant":args.variant,"metrics":report}),flush=True)

if __name__=="__main__":main()
