from __future__ import annotations
import argparse, json, os, resource, subprocess, sys, tracemalloc
from pathlib import Path

STAGES=("python","numpy","package","weights","runtime")
MODEL_SHA="e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54"

def rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("Linux only")

def one(stage,path):
    import hashlib
    initial=rss()
    tracemalloc.start()
    objs={}
    if stage in ("numpy","package","weights","runtime"):
        import numpy
        objs["numpy"]=numpy
    after_numpy=rss()
    if stage in ("package","weights","runtime"):
        import codeccat
        objs["codeccat"]=codeccat
    after_package=rss()
    if stage in ("weights","runtime"):
        from codeccat import load_bundle
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==MODEL_SHA
        objs["bundle"]=load_bundle(Path(path))
    after_weights=rss()
    model_bytes=0
    if "bundle" in objs:
        from codeccat.model_io import _put_stack,_put_model
        arrays={}; meta={}
        bun=objs["bundle"]
        _put_stack(arrays,meta,"baseline",bun.baseline)
        _put_stack(arrays,meta,"full_b",bun.full_b)
        _put_model(arrays,meta,"rl_specialist",bun.rl_specialist)
        model_bytes=sum(getattr(x,"nbytes",0) for x in arrays.values())
    if stage=="runtime":
        from codeccat.runtime import Runtime
        objs["runtime"]=Runtime(objs["bundle"])
    after_runtime=rss()
    cur,peak=tracemalloc.get_traced_memory()
    return dict(stage=stage,initial_kib=initial,after_numpy_kib=after_numpy,
        after_package_kib=after_package,after_weights_kib=after_weights,
        after_runtime_kib=after_runtime,model_numpy_nbytes=model_bytes,
        traced_current_bytes=cur,traced_peak_bytes=peak,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--worker",choices=STAGES)
    ap.add_argument("--out",default="research-results/memory-attribution")
    a=ap.parse_args()
    if a.worker:
        print("MEMORY_STAGE_JSON",json.dumps(one(a.worker,a.model)),flush=True)
        return
    reports={}
    for stage in STAGES:
        completed=subprocess.run([sys.executable,__file__,"--model",a.model,"--worker",stage],capture_output=True,text=True,check=True)
        match=[l.split("MEMORY_STAGE_JSON ",1)[1] for l in completed.stdout.splitlines() if l.startswith("MEMORY_STAGE_JSON ")]
        assert len(match)==1,(stage,completed.stdout,completed.stderr)
        reports[stage]=json.loads(match[0])
    dest=Path(a.out)
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"stages.json").write_text(json.dumps(reports,indent=2))
    print("MEMORY_ATTRIBUTION_START")
    print(json.dumps(reports,indent=2))
    print("MEMORY_ATTRIBUTION_END")
if __name__=="__main__":main()
