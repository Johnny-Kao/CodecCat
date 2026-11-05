"""Isolated-process resource comparison, pinned mixed real bytes + fixed edge cases.

One environment per library, same sample ordering/bytes, same runner.
Metrics are process-based and are *not* a claim of API semantic equivalence:
chardet detects a label, charset-normalizer .best() performs coherence
selection, CodecCat.detect() also strictly decodes and checks ambiguity.
"""
from __future__ import annotations
import argparse,base64,codecs,json,os,platform,resource,statistics,sys,time
from collections import defaultdict
from pathlib import Path

FIXTURES=Path("native-prototype/resource-fixtures.jsonl")
RESULTS=Path("native-prototype/resource-results")

def prepare():
    from vnext_actual_adapter import corpus
    rows=corpus(Path("training"))
    # Deterministic, source-group dispersed selection from real test files;
    # prefer retaining all scarce short cases, plus a long sample.
    shorts=[r for r in rows if len(r[0])<=256]
    longs=[r for r in rows if len(r[0])>256]
    selection=shorts[:80]+longs[::max(1,len(longs)//220)][:220]
    selection.extend([
        (b"hello", "ascii", "edge", -1, ""),
        ("日本語とcafé".encode("utf-8"),"utf-8","edge",-1,""),
        ("café crème".encode("cp1252"),"cp1252","edge",-1,""),
        (("日本語とcafé "*12000).encode("utf-8"),"utf-8","edge",-1,""),
    ])
    FIXTURES.write_text("".join(json.dumps({
        "raw":base64.b64encode(raw).decode("ascii"),"expected":enc,
        "source":"chardet/test-data" if group!="edge" else "fixed-edge"
    })+"\n" for raw,enc,group,_,_ in selection))
    print("RESOURCE_FIXTURES_PREPARED",len(selection),
          json.dumps({"short":len(shorts[:80]),"long":len(selection)-len(shorts[:80])}),flush=True)

def p95(xs):
    v=sorted(xs)
    return v[int((len(v)-1)*.95)]

def main(variant):
    begin=time.perf_counter()
    if variant=="native":
        from codeccat import CodecCat
        imported=time.perf_counter()
        model=CodecCat()
        def detect(raw):
            r=model.detect(raw)
            return r.encoding,r.text
    elif variant=="alpha":
        from codeccat import CodecCat
        imported=time.perf_counter()
        model=CodecCat.from_files("src/codeccat/models/stable23.zip")
        def detect(raw):
            r=model.detect(raw)
            return r.encoding,r.text
    elif variant=="chardet":
        import chardet
        imported=time.perf_counter()
        def detect(raw):
            enc=chardet.detect(raw).get("encoding")
            return enc,None
    elif variant=="charset-normalizer":
        from charset_normalizer import from_bytes
        imported=time.perf_counter()
        def detect(raw):
            match=from_bytes(raw).best()
            return match.encoding if match else None,None
    else:raise ValueError(variant)
    loaded=time.perf_counter()
    import codecs
    records=[json.loads(l) for l in FIXTURES.read_text().splitlines()]
    cases=[(base64.b64decode(x["raw"]),x["expected"],x["source"]) for x in records]
    # A small, equal warmup for each process.
    for raw,_,_ in cases[:5]:detect(raw)
    delays=[];cpu=[];hits=0;counts=defaultdict(lambda:{"n":0,"correct":0})
    for raw,expected,source in cases:
        start=time.perf_counter_ns();pc=time.process_time_ns()
        encoding,text=detect(raw)
        cpu.append((time.process_time_ns()-pc)/1e6)
        delays.append((time.perf_counter_ns()-start)/1e6)
        try:gold=raw.decode(expected,"strict").lstrip("\ufeff")
        except (LookupError,UnicodeError):gold=None
        if text is None and encoding:
            try:text=raw.decode(encoding,"strict").lstrip("\ufeff")
            except (LookupError,UnicodeError):text=None
        # CodecCat returns BOM-strip text by default.
        ok=(gold is not None and text==gold)
        hits+=int(ok)
        group="short<=256" if len(raw)<=256 else "long>256"
        counts[group]["n"]+=1;counts[group]["correct"]+=int(ok)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux ru_maxrss is KiB; macOS returns bytes.
    if sys.platform=="darwin":rss_mib=rss/(1024*1024)
    else:rss_mib=rss/1024
    result={
        "variant":variant,"python":platform.python_version(),"system":platform.platform(),
        "n":len(cases),"correct":hits,
        "accuracy_pct":round(hits/len(cases)*100,3),
        "p50_ms":round(statistics.median(delays),4),
        "p95_ms":round(p95(delays),4),
        "cpu_p50_ms":round(statistics.median(cpu),4),
        "cpu_total_ms":round(sum(cpu),2),
        "wall_total_ms":round(sum(delays),2),
        "import_ms":round((imported-begin)*1e3,3),
        "model_load_ms":round((loaded-imported)*1e3,3),
        "maxrss_mib":round(rss_mib,2),
        "buckets":dict(counts),
        "api_note":"CodecCat detects+strictly decodes; chardet detects label only; charset-normalizer uses .best(); different internal work",
        "corpus_note":"Source-group-distributed real chardet test fixtures + fixed edge cases; not independent accuracy certification"
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS/(variant+".json")).write_text(json.dumps(result,indent=2))
    print("RESOURCE_BENCH_RESULT",json.dumps(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("prepare","alpha","native","chardet","charset-normalizer"),required=True)
    a=p.parse_args()
    if a.variant=="prepare":prepare()
    else:main(a.variant)
