"""Independent external-source diagnostic: train chardet corpus, evaluate uchardet corpus.

No external holdout data enters training. Report byte/text overlap explicitly.
Do not claim independence if content overlap is present.
"""
import argparse,codecs,collections,hashlib,json,sys
from pathlib import Path
from codeccat.vnext_strong_l2 import TrainableDetector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus as training_corpus
from vnext_residual_study import residual_subset

def canonical(label):
    label=label.lower().replace("windows-","cp").replace("utf-16be","utf-16-be").replace("utf-16le","utf-16-le")
    try:return codecs.lookup(label).name
    except LookupError:return None

def main(args):
    import chardet
    from charset_normalizer import from_bytes
    training=training_corpus(Path(args.train))
    source_rows=[(b,y,g) for b,y,g,k,sha in training]
    kept,dropped=residual_subset(source_rows)
    model=TrainableDetector.fit(kept)
    train_hashes={hashlib.sha256(b).hexdigest() for b,_,_ in source_rows}
    train_text_hashes=set()
    for b,y,_ in source_rows:
        try:train_text_hashes.add(hashlib.sha256(b.decode(y).encode()).hexdigest())
        except (UnicodeError,LookupError):pass
    seen=set();trace=[]
    for p in sorted(Path(args.external).glob("test/**/*")):
        if not p.is_file() or p.suffix not in (".txt",".be",".le"):continue
        name=p.stem
        if p.suffix in (".be",".le"):name=p.name.replace(".be","-be").replace(".le","-le")
        truth=canonical(name)
        if not truth:continue
        b=p.read_bytes()[:4096]
        if not b:continue
        try:correct=b.decode(truth,"strict")
        except (UnicodeError,LookupError):continue
        dig=hashlib.sha256(b).hexdigest()
        if dig in seen:continue
        seen.add(dig)
        text_hash=hashlib.sha256(correct.encode()).hexdigest()
        ours=model.rank(b,final=len(b)<4096).encoding
        competitor1=canonical(chardet.detect(b).get("encoding") or "")
        try:
            result=from_bytes(b).best()
            competitor2=canonical(result.encoding if result is not None else "")
        except Exception:competitor2=None
        labels={"ours":ours,"chardet":competitor1,"charset_normalizer":competitor2}
        evals={}
        for k,pred in labels.items():
            try:equiv=(pred is not None and b.decode(pred,"strict")==correct)
            except (UnicodeError,LookupError):equiv=False
            evals[k]={"exact":pred==truth,"equiv":equiv}
        trace.append({"path":p.relative_to(args.external).as_posix(),"truth":truth,
                      "overlap_bytes":dig in train_hashes,"overlap_text":text_hash in train_text_hashes,
                      "predictions":labels,"metrics":evals})
    if not trace:raise RuntimeError("No external labeled corpus records discovered")
    n=len(trace);report={"n":n,"train":len(source_rows),"train_after_residual":len(kept),
         "duplicate_training_bytes":sum(x["overlap_bytes"] for x in trace),
         "duplicate_training_decoded_text":sum(x["overlap_text"] for x in trace),
         "source":"TypesettingTools/uchardet/test @ pinned commit, distinct repository; byte/text overlap tracked",
         "labels":dict(collections.Counter(x["truth"] for x in trace)),
         "all":{},"nonoverlap":{}}
    for category,rows in (("all",trace),("nonoverlap",[x for x in trace if not x["overlap_bytes"] and not x["overlap_text"]])):
        report[category]["n"]=len(rows)
        for name in ("ours","chardet","charset_normalizer"):
            report[category][name]={"exact":sum(x["metrics"][name]["exact"] for x in rows),
                    "equiv":sum(x["metrics"][name]["equiv"] for x in rows)}
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    (dest/"report.json").write_text(json.dumps(report,indent=2))
    (dest/"trace.jsonl").write_text("\n".join(json.dumps(t) for t in trace)+"\n")
    print("EXTERNAL_RESULT_BEGIN");print(json.dumps(report,indent=2));print("EXTERNAL_RESULT_END")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True);p.add_argument("--external",required=True)
    p.add_argument("--output",default="research-results/vnext-external-holdout")
    main(p.parse_args())
