"""Paired exact-label vs decoded-text-equivalence validation.

Source corpus remains competitor-owned and is a diagnostic, not release holdout.
No old detector, no externally trained models.
"""
import argparse,codecs,collections,hashlib,json,sys
from pathlib import Path
from codeccat.vnext_strong_l2 import TrainableDetector

def canonical(x):
    try:return codecs.lookup(x).name
    except (LookupError,TypeError):return None

def corpus(root):
    sys.path.insert(0,str(root/"scripts"))
    from check_test_data import parse_dir_name
    records=[];seen=set()
    for f in sorted(root.glob("*/*")):
        if not f.is_file():continue
        name,_=parse_dir_name(f.parent.name)
        truth=canonical(name)
        if not truth:continue
        b=f.read_bytes()[:4096]
        if not b:continue
        try:b.decode(truth,"strict")
        except (UnicodeError,LookupError):continue
        key=hashlib.sha256(b).hexdigest()
        if key in seen:continue
        seen.add(key)
        group=f.stem.lower().replace("culturax_mc4_","culturax_").replace("culturax_oscar-2301_","culturax_")
        fold=int(hashlib.sha256(group.encode()).hexdigest()[:12],16)%3
        records.append((b,truth,group,fold,key))
    return records

def equivalent(b,reference,candidate):
    if candidate is None:return False
    try:return b.decode(reference,"strict")==b.decode(candidate,"strict")
    except (UnicodeError,LookupError):return False

def main(a):
    rows=corpus(Path(a.fixtures))
    trace=[]
    for fold in range(3):
        training=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
        model=TrainableDetector.fit(training)
        for b,y,g,k,sha in rows:
            if k!=fold:continue
            result=model.rank(b,final=len(b)<4096)
            exact=result.encoding==y
            eq=equivalent(b,y,result.encoding)
            # Does data contain anything which distinguishes truth from common
            # confusing codecs? All decoder-equivalent cases are unidentifiable.
            trace.append(dict(truth=y,pred=result.encoding,fold=k,sha=sha,
                              exact=exact,equiv=eq,
                              covered=y in model.classifier.classes_,
                              ambiguity=bool(result.ambiguous)))
        print("EQUIVALENCE_FOLD",fold,flush=True)
    n=len(trace)
    per=collections.defaultdict(lambda:collections.Counter())
    for x in trace:
        counts=per[x["truth"]]
        counts["n"]+=1
        counts["exact"]+=x["exact"]
        counts["equiv"]+=x["equiv"]
        counts["label_mismatch_same_text"]+=int(x["equiv"] and not x["exact"])
        counts["true_text_error"]+=int(not x["equiv"])
    report={"n":n,"exact":sum(t["exact"] for t in trace),
      "equivalent_decoding":sum(t["equiv"] for t in trace),
      "label_mismatch_same_decoding":sum(t["equiv"] and not t["exact"] for t in trace),
      "different_decoded_text":sum(not t["equiv"] for t in trace),
      "unsupported_ground_truth":sum(not t["covered"] for t in trace),
      "by_label":[dict(label=k,**dict(v)) for k,v in sorted(per.items(),key=lambda kv:-kv[1]["true_text_error"])],
      "dataset":"chardet/test-data grouped 3fold - same fixture source is NOT independent",
      "decision":"Do not claim unlabeled identical decodings can be identified from bytes alone"}
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    (output/"report.json").write_text(json.dumps(report,indent=2))
    (output/"trace.jsonl").write_text("\n".join(json.dumps(t) for t in trace)+"\n")
    print("EQUIVALENCE_RESULT_BEGIN");print(json.dumps(report,indent=2));print("EQUIVALENCE_RESULT_END")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True);p.add_argument("--output",default="research-results/vnext-equivalence");main(p.parse_args())
