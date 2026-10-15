"""Paired accuracy gap audit, competitor baseline and error families.

Same bytes, group-held-out trainable model, chardet/charset-normalizer
external baselines; competitor-owned fixtures => development-only.
"""
import argparse,codecs,collections,hashlib,json,sys
from pathlib import Path
from codeccat.vnext_strong_l2 import TrainableDetector

def canon(x):
    try:return codecs.lookup(x).name if x else None
    except LookupError:return None

def load(root):
    sys.path.insert(0,str(root/"scripts"))
    from check_test_data import parse_dir_name
    out=[];seen=set()
    for f in sorted(root.glob("*/*")):
        if not f.is_file():continue
        enc,_=parse_dir_name(f.parent.name)
        label=canon(enc)
        if label is None:continue
        data=f.read_bytes()[:4096]
        if not data:continue
        try:data.decode(label,"strict")
        except (UnicodeError,LookupError):continue
        sha=hashlib.sha256(data).hexdigest()
        if sha in seen:continue
        seen.add(sha)
        group=f.stem.lower().replace("culturax_mc4_","culturax_").replace("culturax_oscar-2301_","culturax_")
        fold=int(hashlib.sha256(group.encode()).hexdigest()[:12],16)%3
        out.append((data,label,group,fold,sha))
    return out

def main(args):
    import chardet
    from charset_normalizer import from_bytes
    rows=load(Path(args.fixtures))
    trace=[]
    for fold in range(3):
        training=[(b,y,g) for b,y,g,k,_ in rows if k!=fold]
        held=[(b,y,g,h) for b,y,g,k,h in rows if k==fold]
        m=TrainableDetector.fit(training)
        for data,truth,group,sha in held:
            ranked=m.rank(data,final=(len(data)<4096))
            cd=canon(chardet.detect(data).get("encoding"))
            try:
                result=from_bytes(data).best()
                cn=canon(result.encoding if result else None)
            except Exception:cn=None
            trace.append({"sha":sha,"fold":fold,"truth":truth,"group":group,
                          "ours":canon(ranked.encoding),"chardet":cd,"charset_normalizer":cn,
                          "top3":[canon(x) for x in ranked.candidates],
                          "truth_in_top3":truth in ranked.candidates})
        print("PAIRED_FOLD_DONE",fold,flush=True)
    n=len(trace)
    models=("ours","chardet","charset_normalizer")
    summary={name:{"correct":sum(x[name]==x["truth"] for x in trace),
                   "accuracy":round(sum(x[name]==x["truth"] for x in trace)/n,5)}
             for name in models}
    pairs={}
    for name in ("chardet","charset_normalizer"):
        pairs[name]={"ours_win":sum(x["ours"]==x["truth"] and x[name]!=x["truth"] for x in trace),
                     "theirs_win":sum(x["ours"]!=x["truth"] and x[name]==x["truth"] for x in trace)}
    errors=collections.defaultdict(lambda:collections.Counter())
    truth_count=collections.Counter(x["truth"] for x in trace)
    for x in trace:
        if x["ours"]!=x["truth"]:
            errors[x["truth"]]["wrong"]+=1
            errors[x["truth"]]["pred_"+str(x["ours"])]+=1
    families=[]
    for label,n_true in truth_count.most_common():
        err=errors[label]
        families.append({"label":label,"n":n_true,"errors":err["wrong"],
                         "accuracy":round((n_true-err["wrong"])/n_true,4),
                         "common_misclassifications":err.most_common(6)[1:]})
    report={"n":n,"source":"chardet/test-data competitor-owned development-only",
      "metric":"exact normalized Python codec label; ambiguous ASCII and aliases NOT equivalence-adjusted",
      "paired":summary,"wins":pairs,"worst_labels":sorted(families,key=lambda x:(-x["errors"],x["label"]))[:30],
      "largest_groups":families[:30],"top3_oracle":sum(x["truth_in_top3"] for x in trace),
      "limitations":"Chardet trained-on/test-data provenance not independent; source stem groups not guaranteed independent."}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/"paired_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/"trace.jsonl").write_text("\n".join(json.dumps(x) for x in trace)+"\n")
    print("PAIRED_RESULT_BEGIN");print(json.dumps(report,ensure_ascii=False,indent=2));print("PAIRED_RESULT_END")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fixtures",required=True);p.add_argument("--output",default="research-results/vnext-paired-audit");main(p.parse_args())
