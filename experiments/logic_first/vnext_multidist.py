"""Multi-distribution randomized audit: source-stratified, family-balanced,
and pooled-natural fixture samples. Immutable seed and source manifests.
Never train on evaluation fixture repositories.
"""
import argparse,codecs,collections,hashlib,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector

def norm(n):
    try:return codecs.lookup(n).name
    except (LookupError,TypeError):return None

def family(n):
    n=n.lower()
    if n.startswith(("utf-","utf_")):return "unicode"
    if n.startswith(("iso8859","cp12","mac-","mac_","koi8","hp-")):return "single-byte"
    if n.startswith(("gb","big5","shift","euc","iso2022","cp932","cp949")):return "east-asian"
    return "other"

def gather(root,source):
    out=[]
    if source=="uchardet":
        for p in sorted((root/"test").rglob("*")):
            if not p.is_file() or p.suffix not in (".txt",".be",".le"):continue
            code=p.stem if p.suffix==".txt" else p.name.replace(".be","-be").replace(".le","-le")
            label=norm(code.replace("windows-","cp"))
            if label:out.append((p,label))
    elif source=="unicode-suite":
        for p in sorted(root.rglob("DOC*.txt")):
            parts=p.name.split("_")
            if len(parts)>5:
                label=norm(parts[4])
                if label:out.append((p,label))
    return out

def run(a):
    import chardet
    from charset_normalizer import from_bytes
    original=[(b,y,g) for b,y,g,k,_ in corpus(Path(a.train))]
    train,_=residual_subset(original)
    detector=TrainableDetector.fit(train)
    known_b={hashlib.sha256(b).hexdigest() for b,y,g in original}
    known_t=set()
    for b,y,g in original:
        try:known_t.add(hashlib.sha256(b.decode(y).encode()).hexdigest())
        except UnicodeError:pass
    records=[];seen=set()
    for source,root in (("uchardet",Path(a.uchardet)),("unicode-suite",Path(a.unicode_suite))):
        for p,y in gather(root,source):
            b=p.read_bytes()[:4096]
            if not b:continue
            try:
                expected=codecs.getincrementaldecoder(y)("strict").decode(b,final=len(b)<4096)
            except UnicodeError:continue
            bh=hashlib.sha256(b).hexdigest()
            th=hashlib.sha256(expected.encode()).hexdigest()
            if bh in known_b or th in known_t or bh in seen:continue
            seen.add(bh)
            records.append(dict(source=source,filename=str(p.relative_to(root)),label=y,
                                category=family(y),data=b,expected=expected,
                                sample_id=bh))
    if not records:raise RuntimeError("No nonoverlapping external test inputs")
    rng=random.Random(a.seed)
    # Produce two distinct distributions with same source-locked candidates.
    by_stratum=collections.defaultdict(list)
    for r in records:by_stratum[(r["source"],r["category"])].append(r)
    balanced=[]
    for k in sorted(by_stratum):
        rows=by_stratum[k][:]
        rng.shuffle(rows)
        balanced.extend(rows[:min(a.cap_per_stratum,len(rows))])
    rng2=random.Random(a.seed)
    natural=records[:]
    rng2.shuffle(natural)
    natural=natural[:a.natural_limit]
    targets={r["sample_id"]:r for r in balanced+natural}
    scored={}
    for uid,r in targets.items():
        raw=r["data"];y=r["label"];expected=r["expected"]
        cn=from_bytes(raw).best()
        predictions={"ours":norm(detector.rank(raw,final=len(raw)<4096).encoding),
                     "chardet_7":norm(chardet.detect(raw).get("encoding")),
                     "charset_normalizer":norm(cn.encoding if cn else None)}
        metrics={}
        for name,p in predictions.items():
            try:equiv=codecs.getincrementaldecoder(p)("strict").decode(raw,final=len(raw)<4096)==expected
            except (UnicodeError,LookupError,TypeError):equiv=False
            metrics[name]={"exact":p==y,"same_text":equiv,"pred":p}
        scored[uid]={"source":r["source"],"file":r["filename"],"label":y,
                     "family":r["category"],"sha":uid,"metrics":metrics}
    report={"seed":a.seed,"candidate_pool":len(records),
            "strata":{"/".join(k):len(v) for k,v in sorted(by_stratum.items())},
            "distributions":{},"warning":"both distributions reuse the same data sources; not statistically independent; generated text overrepresentation"}
    for title,items in (("balanced_source_family",balanced),("pooled_random",natural)):
        rows=[scored[x["sample_id"]] for x in items]
        report["distributions"][title]={"n":len(rows),"source_counts":dict(collections.Counter(x["source"] for x in rows)),
           "family_counts":dict(collections.Counter(x["family"] for x in rows)),
           "metrics":{name:{"exact":sum(x["metrics"][name]["exact"] for x in rows),
                              "same_text":sum(x["metrics"][name]["same_text"] for x in rows)}
                      for name in ("ours","chardet_7","charset_normalizer")}}
    report["paired_error_diagnostics"]={}
    for title,items in (("balanced_source_family",balanced),("pooled_random",natural)):
        rows=[scored[x["sample_id"]] for x in items]
        diag={}
        for fam_name in sorted(set(x["family"] for x in rows)):
            rr=[x for x in rows if x["family"]==fam_name]
            only_ours=[x for x in rr if x["metrics"]["ours"]["same_text"] and not x["metrics"]["chardet_7"]["same_text"]]
            only_other=[x for x in rr if x["metrics"]["chardet_7"]["same_text"] and not x["metrics"]["ours"]["same_text"]]
            wrong=collections.Counter((x["label"],x["metrics"]["ours"]["pred"]) for x in rr if not x["metrics"]["ours"]["same_text"])
            diag[fam_name]={"n":len(rr),
                "ours_text":sum(x["metrics"]["ours"]["same_text"] for x in rr),
                "chardet_text":sum(x["metrics"]["chardet_7"]["same_text"] for x in rr),
                "only_ours":len(only_ours),"only_chardet":len(only_other),
                "top_ours_errors":[{"truth":a,"pred":b,"n":count} for (a,b),count in wrong.most_common(12)]}
        report["paired_error_diagnostics"][title]=diag
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    (output/"report.json").write_text(json.dumps(report,indent=2))
    (output/"traces.jsonl").write_text("\n".join(json.dumps(x) for x in scored.values())+"\n")
    (output/"selection.json").write_text(json.dumps({k:[x["sample_id"] for x in v] for k,v in
       (("balanced_source_family",balanced),("pooled_random",natural))},indent=2))
    print("MULTIDIST_BEGIN");print(json.dumps(report,indent=2));print("MULTIDIST_END")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True);p.add_argument("--uchardet",required=True)
    p.add_argument("--unicode-suite",required=True)
    p.add_argument("--seed",type=int,default=20261015);p.add_argument("--cap-per-stratum",type=int,default=50)
    p.add_argument("--natural-limit",type=int,default=350)
    p.add_argument("--output",default="research-results/vnext-multidist")
    run(p.parse_args())
