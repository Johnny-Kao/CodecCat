"""Third-source diagnostic against upstream charset-normalizer's labeled fixtures.

Never train on charset-normalizer or uchardet samples. Labels are taken from
the upstream test_full_detection.py at pinned SHA, not predictions.
"""
import argparse,codecs,hashlib,json,sys
from pathlib import Path
from codeccat.vnext_strong_l2 import TrainableDetector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset

FIXTURES={
"sample-arabic-1.txt":"cp1256",
"sample-french-1.txt":"cp1252",
"sample-arabic.txt":"utf-8",
"sample-russian-3.txt":"utf-8",
"sample-french.txt":"utf-8",
"sample-chinese.txt":"big5",
"sample-greek.txt":"cp1253",
"sample-greek-2.txt":"cp1253",
"sample-hebrew-2.txt":"cp1255",
"sample-hebrew-3.txt":"cp1255",
"sample-bulgarian.txt":"utf-8",
"sample-english.bom.txt":"utf-8",
"sample-spanish.txt":"utf-8",
"sample-korean.txt":"cp949",
"sample-turkish.txt":"cp1254",
"sample-russian-2.txt":"utf-8",
"sample-russian.txt":"mac-cyrillic",
"sample-polish.txt":"utf-8",
}
def canon(v):
    if v is None:return None
    try:return codecs.lookup(v).name
    except LookupError:return None

def run(a):
    import chardet
    from charset_normalizer import from_bytes
    source=[(b,y,g) for b,y,g,k,_ in corpus(Path(a.train))]
    chosen,_=residual_subset(source)
    detector=TrainableDetector.fit(chosen)
    known_bytes={hashlib.sha256(b).hexdigest() for b,_,_ in source}
    known_texts=set()
    for b,y,_ in source:
        try:known_texts.add(hashlib.sha256(b.decode(y).encode("utf-8")).hexdigest())
        except UnicodeError:pass
    trace=[]
    for name,source_label in FIXTURES.items():
        raw=(Path(a.external)/"data"/name).read_bytes()[:4096]
        truth=canon(source_label)
        if truth is None:raise ValueError(name)
        text=raw.decode(truth,"strict")
        checksum=hashlib.sha256(raw).hexdigest()
        textsum=hashlib.sha256(text.encode("utf-8")).hexdigest()
        res=from_bytes(raw).best()
        names={"ours":canon(detector.rank(raw,final=len(raw)<4096).encoding),
               "chardet":canon(chardet.detect(raw).get("encoding")),
               "charset_normalizer":canon(res.encoding if res is not None else None)}
        scores={}
        for brand,pred in names.items():
            try:correct_text=pred is not None and raw.decode(pred)==text
            except (UnicodeError,LookupError):correct_text=False
            scores[brand]={"label_correct":pred==truth,"text_correct":correct_text}
        trace.append({"file":name,"truth":truth,"predictions":names,"scores":scores,
                      "same_train_bytes":checksum in known_bytes,
                      "same_train_text":textsum in known_texts})
    clean=[r for r in trace if not r["same_train_bytes"] and not r["same_train_text"]]
    report={"n_total":len(trace),"n_nonoverlap":len(clean),
            "duplicate_bytes":sum(x["same_train_bytes"] for x in trace),
            "duplicate_text":sum(x["same_train_text"] for x in trace),
            "source":"Ousret/charset_normalizer/test_full_detection.py pinned 4b335594aee803e04b62e8ffa1a5a2b7ebd4d33c",
            "all":{},"nonoverlap":{}}
    for k,rows in (("all",trace),("nonoverlap",clean)):
        report[k]["n"]=len(rows)
        for brand in ("ours","chardet","charset_normalizer"):
            report[k][brand]={
                "exact":sum(r["scores"][brand]["label_correct"] for r in rows),
                "same_text":sum(r["scores"][brand]["text_correct"] for r in rows)}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"trace.jsonl").write_text("\n".join(json.dumps(r) for r in trace)+"\n")
    print("THIRD_SOURCE_RESULT_START")
    print(json.dumps(report,indent=2))
    print("THIRD_SOURCE_RESULT_END")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True);p.add_argument("--external",required=True)
    p.add_argument("--output",default="research-results/vnext-third-source")
    run(p.parse_args())
