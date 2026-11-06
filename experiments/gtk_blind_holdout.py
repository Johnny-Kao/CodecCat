"""Pre-registered source-disjoint GTK PO holdout, three engines, five deterministic folds.

Input-source commit, extraction and labels fixed BEFORE any results inspected.
Not a claim of all-domain unbiased accuracy: upstream PO projects share genre.
"""
from __future__ import annotations
import codecs,hashlib,json,re,statistics,time,resource,sys,platform
from pathlib import Path
from collections import defaultdict,Counter
import chardet,codeccat
from charset_normalizer import from_bytes
ROOT=Path("blind-source/po")
OUT=Path("research-results/gtk-blind")
COMMIT="7c578d79eb8347b14a1717bfcc0084674a43c562"
SEED="20261106-gtk-independent-v1"

def norm(label):
    if not label: return None
    try:return codecs.lookup(label).name
    except (LookupError, TypeError):return None
def decoded(raw, label):
    enc=norm(label)
    if not enc:return None
    try:return raw.decode(enc,"strict").lstrip("\ufeff")
    except (UnicodeError,LookupError):return None

def cases():
    entries=[]
    for p in sorted(ROOT.glob("*.po")):
        raw=p.read_bytes()
        match=re.search(br"charset=([A-Za-z0-9_.+-]+)",raw,re.I)
        if not match:continue
        label=norm(match.group(1).decode("ascii"))
        if not label:continue
        # Hide encoding metadata from detection.
        raw=re.sub(br"(?i)(charset=)[A-Za-z0-9_.+-]+",lambda m:m.group(1)+b"?"*(len(m.group(0))-len(m.group(1))),raw)
        # Same 3 windows per original source document, but each document
        # belongs entirely to exactly one fold; no cross-fold leakage.
        for win,off in enumerate([0,len(raw)//3,2*len(raw)//3]):
            sample=raw[off:off+8192]
            for _ in range(8):
                truth=decoded(sample,label)
                if truth is not None:break
                if win>0 and sample:sample=sample[1:]
                if sample:sample=sample[:-1]
            else:continue
            if len(sample)<128 or sample.isascii():continue
            entries.append((p.name,win,sample,label,truth))
    seen=set()
    rows=[]
    for name,win,raw,label,truth in entries:
        fingerprint=hashlib.sha256(raw).hexdigest()
        if fingerprint in seen:continue
        seen.add(fingerprint)
        fold=int(hashlib.sha256((SEED+":"+name).encode()).hexdigest(),16)%5
        rows.append({"file":name,"window":win,"bytes":raw,"truth_label":label,
                     "truth_text":truth,"sha256":fingerprint,"fold":fold+1})
    assert len(rows)>=150 and len(set(r["file"] for r in rows))>=50, (len(rows),len(set(r["file"] for r in rows)))
    return rows

def predict(engine,raw,detector):
    if engine=="codeccat":
        r=detector.detect(raw)
        return norm(r.encoding),r.text
    if engine=="chardet":
        enc=norm(chardet.detect(raw).get("encoding"))
        return enc,decoded(raw,enc)
    obj=from_bytes(raw).best()
    enc=norm(obj.encoding) if obj else None
    return enc,decoded(raw,enc)

def main():
    import importlib.metadata
    assert importlib.metadata.version("codeccat")=="0.1.0a4"
    version={"codeccat":"0.1.0a4","chardet":importlib.metadata.version("chardet"),
             "charset_normalizer":importlib.metadata.version("charset-normalizer")}
    rows=cases()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=[{"source":r["file"],"window":r["window"],"sha256":r["sha256"],
               "truth_label":r["truth_label"],"fold":r["fold"]} for r in rows]
    (OUT/"manifest.json").write_text(json.dumps({"commit":COMMIT,"seed":SEED,"rows":manifest},indent=2)+"\n")
    data=[]
    for engine in ("codeccat","chardet","charset_normalizer"):
        detector=codeccat.CodecCat() if engine=="codeccat" else None
        for row in rows[:12]:predict(engine,row["bytes"],detector)
        for row in rows:
            b=time.process_time_ns()
            w=time.perf_counter_ns()
            enc,txt=predict(engine,row["bytes"],detector)
            wall=(time.perf_counter_ns()-w)/1e6
            cpu=(time.process_time_ns()-b)/1e6
            data.append({"engine":engine,"fold":row["fold"],"source":row["file"],"window":row["window"],
                         "truth_label":row["truth_label"],"guess":enc,
                         "correct_text":txt==row["truth_text"],"correct_label":enc==row["truth_label"],
                         "wall_ms":wall,"cpu_ms":cpu,"sha256":row["sha256"]})
        print("GTK_HOLDOUT_PROGRESS",engine,flush=True)
    result={}
    for engine in ("codeccat","chardet","charset_normalizer"):
        e=[r for r in data if r["engine"]==engine]
        result[engine]={
            "n":len(e),"correct_text":sum(x["correct_text"] for x in e),
            "correct_label":sum(x["correct_label"] for x in e),
            "cpu_ms":round(sum(x["cpu_ms"] for x in e),2),
            "p50_wall_ms":round(statistics.median(x["wall_ms"] for x in e),3),
            "p95_wall_ms":round(sorted(x["wall_ms"] for x in e)[int(.95*(len(e)-1))],3),
            "source_counts":dict(Counter(x["truth_label"] for x in e)),
            "by_fold":{str(k):{"n":sum(x["fold"]==k for x in e),
                              "correct_text":sum(x["correct_text"] for x in e if x["fold"]==k)}
                       for k in range(1,6)},
            "by_codec":{name:{"n":sum(x["truth_label"]==name for x in e),
                              "correct_text":sum(x["correct_text"] for x in e if x["truth_label"]==name)}
                        for name in sorted(set(x["truth_label"] for x in e))}
        }
    payload={"source":"GNOME/gtk","source_commit":COMMIT,"seed":SEED,
             "model":"published v0.1.0a4, pinned identical source through release SHA256",
             "python":platform.python_version(),"versions":version,
             "results":result,
             "caveats":["source-disjoint, not demonstrated near-duplicate-free",
                        "PO genre only, related to previously studied Vim/Git/GLib",
                        "five deterministic document-level folds; not trained on any fold",
                        "one-run timing; different engine APIs"]}
    (OUT/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    (OUT/"per-case.json").write_text(json.dumps(data,indent=2)+"\n")
    print("GTK_HOLDOUT_RESULT",json.dumps(payload),flush=True)
if __name__=="__main__":main()
