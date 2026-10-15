"""Gate 1/5: one-shot paired logic-first architecture tournament.

All six crawls are research-exposed. Labels are WARC declarations. No model tuning,
no release claim, no independent holdout, and no mutation to frozen R21.
"""
import argparse,hashlib,json,random,resource,statistics,time
from collections import Counter,defaultdict
from pathlib import Path
import chardet,charset_normalizer
from charset_normalizer import from_bytes
from codeccat import load_bundle
from codeccat.runtime import Runtime,_signals,family
from codeccat.logic_first import _validate
from h2_fused_candidate_mask_ab import collect,normalize,rank_masked
from h14_japanese_fastpath import fast as h14

SHA="e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54"
CRAWLS=("CC-MAIN-2024-46","CC-MAIN-2025-13","CC-MAIN-2025-51",
        "CC-MAIN-2026-08","CC-MAIN-2026-12","CC-MAIN-2026-39")
CODECS={"utf-8":"utf-8","shift-jis":"shift_jis","euc-jp":"euc_jp",
        "big5":"big5","gb18030":"gb18030"}
POLICIES=("r21","h14","structural_late","strict_late",
          "japanese_contrast","h14_strict","structural_prescore")
COMPETITORS=("chardet_7_6","charset_normalizer_3_5")
SEED=20261015

def decoded(data,codec):
    # Common-Crawl collector may provide 4096-byte prefixes cut mid-character.
    head=data[:4096]
    if not head:return ""
    for trim in range(4):
        if len(head)<=trim:break
        try:return head[:len(head)-trim].decode(codec,"strict")
        except UnicodeDecodeError:pass
    return None

def possible(data,label,mode):
    if label not in CODECS:return True
    if mode=="structure":
        return _validate(data[:4096],label).status!="invalid"
    return decoded(data,CODECS[label]) is not None

def late(data,rank,mode):
    if possible(data,rank[0],mode):return rank[0]
    for alternative in rank[1:4]:
        if possible(data,alternative,mode):return alternative
    return rank[0]

def kana(text):
    if text is None:return 0
    return sum("\u3040"<=c<="\u309f" or "\u30a0"<=c<="\u30ff" for c in text)

def jp_contrast(data,first):
    if _signals(data).route!="RH":return first
    sj=kana(decoded(data,"shift_jis"))
    ej=kana(decoded(data,"euc_jp"))
    if sj>=2 and sj>=ej+2:return "shift-jis"
    if ej>=2 and ej>=sj+2:return "euc-jp"
    return first

def predict(rt,data,rank,policy):
    first=rank[0]
    if policy=="r21":return first
    if policy=="h14":return h14(data,first)[0]
    if policy=="structural_late":return late(data,rank,"structure")
    if policy=="strict_late":return late(data,rank,"strict")
    if policy=="japanese_contrast":return jp_contrast(data,first)
    if policy=="h14_strict":
        h=h14(data,first)[0]
        return late(data,(h,)+tuple(c for c in rank if c!=h),"strict")
    if policy=="structural_prescore":return rank_masked(rt,data)[0]
    raise ValueError(policy)

def metrics(rows,key):
    n=len(rows)
    hit=sum(r["pred"][key]==r["truth"] for r in rows)
    bhit=sum(r["pred"]["r21"]==r["truth"] for r in rows)
    return {"n":n,"hits":hit,"acc":round(hit/max(n,1),6),
            "delta_hits":hit-bhit,
            "helped":sum(r["pred"][key]==r["truth"] and r["pred"]["r21"]!=r["truth"] for r in rows),
            "harmed":sum(r["pred"][key]!=r["truth"] and r["pred"]["r21"]==r["truth"] for r in rows),
            "changed":sum(r["pred"][key]!=r["pred"]["r21"] for r in rows)}

def main(model,out):
    if hashlib.sha256(Path(model).read_bytes()).hexdigest()!=SHA:
        raise ValueError("immutable R21 SHA mismatch")
    rt=Runtime(load_bundle(model))
    seen=set();rows=[];truth_struct=Counter();per_crawl={}
    for crawl in CRAWLS:
        samples,_,_=collect(crawl)
        added=0
        for item in samples:
            data=item["data"];digest=hashlib.sha256(data).hexdigest()
            if digest in seen:continue
            seen.add(digest);added+=1
            truth=normalize(item["label"])
            rank=rt.rank(data)
            preds={name:predict(rt,data,rank,name) for name in POLICIES}
            ch=chardet.detect(data).get("encoding")
            cn=from_bytes(data).best()
            preds["chardet_7_6"]=normalize(ch) if ch else None
            preds["charset_normalizer_3_5"]=normalize(cn.encoding) if cn else None
            if truth in CODECS:
                ev=_validate(data[:4096],truth).status
                truth_struct[truth+":"+ev]+=1
            rows.append({"crawl":crawl,"sha":digest,"truth":truth,
                         "family":family(truth),"route":_signals(data).route,
                         "bytes":len(data),"top3":list(rank[:3]),"pred":preds,
                         "_data":data})
        per_crawl[crawl]=added
        print("GATE1_COLLECTED",crawl,added,flush=True)
    if len(rows)<500:raise ValueError(f"too few samples: {len(rows)}")
    names=POLICIES+COMPETITORS
    total={name:metrics(rows,name) for name in names}
    splits={}
    for key in ("crawl","family","route"):
        spl={}
        for value in sorted(set(r[key] for r in rows)):
            segment=[r for r in rows if r[key]==value]
            spl[value]={name:metrics(segment,name) for name in names}
        splits[key]=spl
    # Warm end-to-end timing on one fixed 384-sample paired subset.
    rng=random.Random(SEED);subset=rng.sample(rows,min(384,len(rows)))
    perf={}
    for name in names:
        batches=[]
        for repetition in range(3):
            seq=subset if repetition%2==0 else list(reversed(subset))
            start=time.perf_counter_ns()
            for row in seq:
                data=row["_data"]
                if name=="chardet_7_6":chardet.detect(data)
                elif name=="charset_normalizer_3_5":from_bytes(data).best()
                else:
                    rank=rt.rank(data)
                    predict(rt,data,rank,name)
            batches.append((time.perf_counter_ns()-start)/len(seq))
        perf[name]={"median_ns":round(statistics.median(batches)),
                    "samples_per_repeat":len(subset),"repeats":3}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report={"gate":1,"total_gates":5,"kind":"development-only architecture direction",
            "data_warning":"Research-exposed crawls, weak declared labels, not untouched holdout",
            "model_sha256":SHA,"seed":SEED,
            "versions":{"chardet":chardet.__version__,
                        "charset_normalizer":charset_normalizer.__version__},
            "collection":per_crawl,"n":len(rows),"total":total,"split":splits,
            "declared_truth_structure":dict(truth_struct),
            "performance":perf,
            "mixed_process_peak_rss_mib_linux":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
            "limitations":"Timing is median per-sample throughput (not p95). Mixed-process RSS not comparable per library."}
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"paired_trace.jsonl").write_text("\n".join(json.dumps(
         {k:v for k,v in row.items() if k!="_data"},sort_keys=True) for row in rows)+"\n")
    print("GATE1_RESULT_START",flush=True)
    print(json.dumps({key:report[key] for key in
          ("versions","collection","n","total","declared_truth_structure","performance")},indent=2),flush=True)
    print("GATE1_RESULT_END",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--out",default="research-results/vnext-gate1")
    a=p.parse_args()
    main(a.model,a.out)
