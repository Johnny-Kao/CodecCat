"""Frozen external corpus diagnostic; not a training input nor tuned selection.

Source: UnicodeTestSuite generated from UDHR texts (synthetic re-encodings).
Stratified deterministic selection by document and encoding, overlap filtered.
"""
import argparse,codecs,csv,hashlib,json,collections,sys
from pathlib import Path
from codeccat.vnext_strong_l2 import TrainableDetector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus as train_corpus
from vnext_residual_study import residual_subset
def canon(label):
    try:return codecs.lookup(label).name
    except (LookupError,TypeError):return None
def main(a):
    import chardet
    from charset_normalizer import from_bytes
    source=[(b,y,g) for b,y,g,k,_ in train_corpus(Path(a.train))]
    kept,_=residual_subset(source)
    model=TrainableDetector.fit(kept)
    oldb={hashlib.sha256(b).hexdigest() for b,_,_ in source}
    oldt=set()
    for b,y,_ in source:
        try:oldt.add(hashlib.sha256(b.decode(y).encode()).hexdigest())
        except (UnicodeError,LookupError):pass
    files=[]
    for p in Path(a.external).rglob("DOC*.txt"):
        name=p.name.split("_")
        if len(name)<6:continue
        truth=canon(name[4])
        if truth is None:continue
        files.append((p,truth))
    # Hash-stable sampled subset, a fixed sample selection uninformed by any
    # detector score; cap to keep public CI time bounded.
    files.sort(key=lambda t:hashlib.sha256(str(t[0].relative_to(a.external)).encode()).hexdigest())
    files=files[:a.limit]
    trace=[]
    for p,y in files:
        b=p.read_bytes()[:4096]
        if not b:continue
        try:
            decoder=codecs.getincrementaldecoder(y)("strict")
            target=decoder.decode(b,final=len(b)<4096)
        except UnicodeError:continue
        bytehash=hashlib.sha256(b).hexdigest()
        texthash=hashlib.sha256(target.encode()).hexdigest()
        if bytehash in oldb or texthash in oldt:continue
        ours=canon(model.rank(b,final=len(b)<4096).encoding)
        base=canon(chardet.detect(b).get("encoding"))
        cn=from_bytes(b).best()
        other=canon(cn.encoding if cn is not None else None)
        result={}
        for name,guess in (("ours",ours),("chardet_7",base),("charset_normalizer",other)):
            try:
                dec=codecs.getincrementaldecoder(guess)("strict")
                eq=dec.decode(b,final=len(b)<4096)==target
            except (UnicodeError,LookupError,TypeError):eq=False
            result[name]={"exact":guess==y,"same_text":eq,"guess":guess}
        trace.append({"filename":str(p.relative_to(a.external)),"truth":y,**result})
    out={"source":"amrali-eg/UnicodeTestSuite 80684bd405505b069cfec4e07493c8713a26f41a generated UDHR transcodings",
         "sample_limit":a.limit,"n":len(trace),"note":"synthetic encoding corpus, no tuning, correlated by original UDHR document; NOT naturally occurring file distribution",
         "metrics":{key:{"exact":sum(v[key]["exact"] for v in trace),"same_text":sum(v[key]["same_text"] for v in trace)}
                    for key in ("ours","chardet_7","charset_normalizer")},
         "labels":dict(collections.Counter(v["truth"] for v in trace))}
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    (dest/"report.json").write_text(json.dumps(out,indent=2))
    (dest/"trace.jsonl").write_text("\n".join(map(json.dumps,trace))+"\n")
    print("UDHR_RESULT_BEGIN");print(json.dumps(out,indent=2));print("UDHR_RESULT_END")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--train",required=True)
    ap.add_argument("--external",required=True);ap.add_argument("--limit",type=int,default=300)
    ap.add_argument("--output",default="research-results/vnext-udhr")
    main(ap.parse_args())
