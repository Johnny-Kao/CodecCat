"""External multi-encoding source-disjoint held-out sample evaluation.

Source: BYVoid/uchardet, test/ data pinned to a commit before evaluation.
No model tuning. Multiple windows per original document stay in a single fold.
Unsupported Python encodings are listed and excluded, never silently relabeled.
"""
from __future__ import annotations
import codecs,hashlib,importlib.metadata,json,platform,statistics,time
from pathlib import Path
from collections import Counter,defaultdict
import codeccat,chardet
from charset_normalizer import from_bytes
ROOT=Path("external-encoding-source/test")
OUT=Path("research-results/uchardet-holdout")
SOURCE_REV="4e685757780cb3c652fc6c9ec759f62888969ec9"
SEED="uchardet-heldout-v1-20261106"
def norm(label):
    if not label:return None
    try:return codecs.lookup(label).name
    except (TypeError,LookupError):return None
def text(raw,enc):
    if not enc:return None
    try:return raw.decode(enc,"strict").lstrip("\ufeff")
    except (UnicodeError,LookupError):return None
def source_samples():
    rows=[];skips=[];seen=set()
    for path in sorted(ROOT.glob("*/*")):
        if not path.is_file() or path.name=="uchardet-tests.c":continue
        enc=path.stem.lower().replace("utf-16be","utf-16-be").replace("utf-16le","utf-16-le").replace("utf-32be","utf-32-be").replace("utf-32le","utf-32-le")
        if path.suffix.lower() in (".le",".be"):
            enc=path.stem+"-"+path.suffix[1:]
        if path.suffix.lower() not in (".txt",".smi",".le",".be"):continue
        canon=norm(enc)
        if not canon:
            skips.append({"path":path.relative_to(ROOT).as_posix(),"encoding":enc,"reason":"unsupported codec"})
            continue
        raw=path.read_bytes()
        if text(raw,canon) is None:
            skips.append({"path":path.relative_to(ROOT).as_posix(),"encoding":enc,"reason":"invalid ground-truth decode"})
            continue
        for take in (256,4096,len(raw)):
            if take==len(raw):piece=raw
            else:
                piece=raw[:min(len(raw),take)]
                # Drop truncated terminal characters; preserve initial BOM.
                while piece and text(piece,canon) is None:piece=piece[:-1]
            if len(piece)<32:continue
            fp=hashlib.sha256(piece).hexdigest()
            if fp in seen:continue
            seen.add(fp)
            rel=path.relative_to(ROOT).as_posix()
            fold=int(hashlib.sha256((SEED+rel).encode()).hexdigest(),16)%5+1
            rows.append({"source":rel,"raw":piece,"truth":canon,"expected":text(piece,canon),
                         "size":len(piece),"sha256":fp,"fold":fold})
    assert len(rows)>=75,(len(rows),skips)
    assert len({r["truth"] for r in rows})>=20,"encoding diversity insufficient"
    return rows,skips
def predict(engine,raw,det):
    if engine=="codeccat":
        r=det.detect(raw)
        return norm(r.encoding),r.text
    if engine=="chardet":
        enc=norm(chardet.detect(raw).get("encoding"))
        return enc,text(raw,enc)
    obj=from_bytes(raw).best()
    enc=norm(obj.encoding) if obj else None
    return enc,text(raw,enc)
def main():
    assert importlib.metadata.version("codeccat")=="0.1.0a4"
    rows,skips=source_samples()
    OUT.mkdir(parents=True,exist_ok=True)
    metrics={};cases=[]
    for eng in ("codeccat","chardet","charset_normalizer"):
        det=codeccat.CodecCat() if eng=="codeccat" else None
        for r in rows[:10]:predict(eng,r["raw"],det)
        durations=[];cpu=[]
        for r in rows:
            st=time.perf_counter_ns();ct=time.process_time_ns()
            guess,decoded=predict(eng,r["raw"],det)
            elapsed=(time.perf_counter_ns()-st)/1e6;cpu_elapsed=(time.process_time_ns()-ct)/1e6
            durations.append(elapsed);cpu.append(cpu_elapsed)
            cases.append({"engine":eng,"sha256":r["sha256"],"file":r["source"],
                          "fold":r["fold"],"size":r["size"],"truth":r["truth"],"guess":guess,
                          "text_correct":decoded==r["expected"],"label_correct":guess==r["truth"]})
        e=cases[-len(rows):]
        metrics[eng]={"n":len(rows),"correct_text":sum(z["text_correct"] for z in e),
                      "correct_label":sum(z["label_correct"] for z in e),
                      "cpu_ms":round(sum(cpu),3),
                      "p50_ms":round(statistics.median(durations),3),
                      "p95_ms":round(sorted(durations)[int(.95*(len(durations)-1))],3),
                      "by_fold":{str(i):{"n":sum(z["fold"]==i for z in e),
                                          "correct_text":sum(z["text_correct"] for z in e if z["fold"]==i)} for i in range(1,6)},
                      "by_codec":{n:{"n":sum(z["truth"]==n for z in e),
                                     "correct_text":sum(z["text_correct"] for z in e if z["truth"]==n)}
                                  for n in sorted({z["truth"] for z in e})}}
    out={"source":"BYVoid/uchardet/test","source_commit":SOURCE_REV,"seed":SEED,
         "cases":len(rows),"original_files":len({r["source"] for r in rows}),
         "encodings":dict(Counter(r["truth"] for r in rows)),
         "skipped_files":skips,"results":metrics,
         "versions":{"codeccat":importlib.metadata.version("codeccat"),
                     "chardet":importlib.metadata.version("chardet"),
                     "charset-normalizer":importlib.metadata.version("charset-normalizer")},
         "caveats":["source-disjoint is not proof of semantic independence from training",
                    "upstream labeled test assets, possible prior reuse in wider OSS corpus",
                    "multiple length windows correlated by document",
                    "one-run timing; not a representative population distribution"]}
    (OUT/"summary.json").write_text(json.dumps(out,indent=2)+"\n")
    (OUT/"per-case.json").write_text(json.dumps(cases,indent=2)+"\n")
    (OUT/"manifest.json").write_text(json.dumps([{"file":r["source"],"sha256":r["sha256"],"truth":r["truth"],"fold":r["fold"],"size":r["size"]} for r in rows],indent=2)+"\n")
    print("MULTICODEC_HOLDOUT_RESULT",json.dumps(out),flush=True)
if __name__=="__main__":main()
