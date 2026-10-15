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
    # Two independent fits on immutable ordered rows must produce identical models.
    repeat=TrainableDetector.fit(kept)
    import numpy as np
    if model.vectorizer.vocabulary_ != repeat.vectorizer.vocabulary_:
        raise AssertionError("non-deterministic vocabulary")
    if not np.array_equal(model.classifier.coef_,repeat.classifier.coef_):
        raise AssertionError("non-deterministic coefficients")
    if not np.array_equal(model.classifier.intercept_,repeat.classifier.intercept_):
        raise AssertionError("non-deterministic intercept")
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
        if ours!=repeat.rank(b,final=len(b)<4096).encoding:
            raise AssertionError("repeat training prediction mismatch")
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
         "deterministic_retrain_equal":True,
         "model_fingerprint":model.fingerprint,
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
    # Evidence ledger: focus on real decoded-text errors, not encoding aliases.
    clean=[x for x in trace if not x["overlap_bytes"] and not x["overlap_text"]]
    per_label=collections.defaultdict(lambda:collections.Counter())
    examples=[]
    for item in clean:
        truth=item["truth"]
        c=per_label[truth]
        c["n"]+=1
        for name in ("ours","chardet","charset_normalizer"):
            c[name+"_exact"]+=int(item["metrics"][name]["exact"])
            c[name+"_text"]+=int(item["metrics"][name]["equiv"])
        if not item["metrics"]["ours"]["equiv"]:
            examples.append({"path":item["path"],"truth":truth,
                             "ours":item["predictions"]["ours"],
                             "charset_normalizer":item["predictions"]["charset_normalizer"],
                             "competitor_text_correct":item["metrics"]["charset_normalizer"]["equiv"]})
    report["nonoverlap_by_label"]=[
        {"label":name,**dict(count)} for name,count in sorted(per_label.items(),
                    key=lambda kv:(-(kv[1]["n"]-kv[1]["ours_text"]),kv[0]))]
    report["actual_text_failure_examples"]=examples
    report["nonoverlap_head_to_head"]={
        "ours_only_text_correct":sum(x["metrics"]["ours"]["equiv"] and not x["metrics"]["charset_normalizer"]["equiv"] for x in clean),
        "competitor_only_text_correct":sum(not x["metrics"]["ours"]["equiv"] and x["metrics"]["charset_normalizer"]["equiv"] for x in clean)}
    report["acceptance_note"]="Small external diagnostic; not a fixed accuracy acceptance criterion. Never tune model on these records."
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    (dest/"report.json").write_text(json.dumps(report,indent=2))
    (dest/"trace.jsonl").write_text("\n".join(json.dumps(t) for t in trace)+"\n")
    print("EXTERNAL_RESULT_BEGIN");print(json.dumps(report,indent=2));print("EXTERNAL_RESULT_END")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True);p.add_argument("--external",required=True)
    p.add_argument("--output",default="research-results/vnext-external-holdout")
    main(p.parse_args())
