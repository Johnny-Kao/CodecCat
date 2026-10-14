"""H13: predeclared Shift-JIS decode evidence, one-shot frozen-R21 A/B.

No training, no threshold sweep. Fixed rule: RH only, strict shift_jis decode of
first 4096 bytes (allow <=3 trailing bytes trimmed due to truncation), >=2 kana,
and original R21 prediction not already shift-jis. Detect harm by crawl/family.
"""
import argparse,hashlib,json,statistics,time
from collections import Counter,defaultdict
from pathlib import Path
import chardet
from charset_normalizer import from_bytes
from codeccat import load_bundle
from codeccat.runtime import Runtime,_signals,family
from h2_fused_candidate_mask_ab import collect,normalize
from h12_japanese_decoder_evidence import decode_features

CRAWLS=("CC-MAIN-2024-22","CC-MAIN-2024-46","CC-MAIN-2024-51",
        "CC-MAIN-2025-13","CC-MAIN-2025-38","CC-MAIN-2025-51","CC-MAIN-2026-04",
        "CC-MAIN-2026-08","CC-MAIN-2026-12","CC-MAIN-2026-17",
        "CC-MAIN-2026-21","CC-MAIN-2026-34","CC-MAIN-2026-39")
SHA="e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54"
def candidate(data,baseline):
    if baseline=="shift-jis" or _signals(data).route!="RH":return baseline,False
    # Strict validity by itself is insufficient; require Japanese phonetic characters.
    evidence=decode_features(data)["shift-jis"]
    if evidence["strict_possible"] and evidence["hira"]+evidence["kata"]>=2:
        return "shift-jis",True
    return baseline,False
def summary(rows):
    n=len(rows);hit=lambda fld:sum(r[fld]==r["truth"] for r in rows)
    return {"n":n,"r21":hit("r21"),"candidate":hit("candidate"),
        "chardet7":hit("chardet7"),"charset_normalizer":hit("charset_normalizer"),
        "changed":sum(r["r21"]!=r["candidate"] for r in rows),
        "beneficial":sum(r["r21"]!=r["truth"] and r["candidate"]==r["truth"] for r in rows),
        "harmful":sum(r["r21"]==r["truth"] and r["candidate"]!=r["truth"] for r in rows),
        "false_positive_non_japanese":sum(r["truth"] not in ("shift-jis","euc-jp")
                                         and r["candidate"]=="shift-jis" and r["r21"]!="shift-jis" for r in rows)}
def main(a):
    assert hashlib.sha256(Path(a.model).read_bytes()).hexdigest()==SHA
    rt=Runtime(load_bundle(a.model));rows=[];seen=set();bytes_pool=[]
    for crawl in CRAWLS:
        samples,_,_=collect(crawl)
        for s in samples:
            data=s["data"];key=hashlib.sha256(data).hexdigest()
            if key in seen:continue
            seen.add(key)
            truth=normalize(s["label"])
            r21=rt.rank(data)[0]
            alt,changed=candidate(data,r21)
            ch=chardet.detect(data).get("encoding")
            cn=from_bytes(data).best()
            rows.append({"crawl":crawl,"family":family(truth),"truth":truth,
                         "route":_signals(data).route,"r21":r21,"candidate":alt,
                         "chardet7":normalize(ch) if ch else None,
                         "charset_normalizer":normalize(cn.encoding) if cn else None,
                         "changed":changed,"sha":key,"bytes":len(data)})
            bytes_pool.append(data)
        print("H13_COLLECTED",crawl,len(samples),flush=True)
    # Inference-only, compare same sample bytes and alternating traversal order.
    perf={}
    for policy in ("r21","candidate"):
        times=[]
        for repeat in range(3):
            data_order=bytes_pool if repeat%2==0 else list(reversed(bytes_pool))
            start=time.perf_counter_ns()
            for data in data_order:
                pred=rt.rank(data)[0]
                if policy=="candidate":candidate(data,pred)
            times.append((time.perf_counter_ns()-start)/len(data_order))
        perf[policy]={"median_ns":statistics.median(times)}
    groups=defaultdict(list)
    for r in rows:groups[r["crawl"]].append(r)
    result={"rule":"RH strict Shift-JIS decode first 4096 bytes allowing <=3 suffix bytes removed; >=2 kana; override to Shift-JIS",
            "candidate_not_trained":True,"no_frozen_model_change":True,
            "data_warning":"All crawls are research-exposed; not an untouched holdout; labels are declarations",
            "total":summary(rows),
            "per_crawl":{k:summary(v) for k,v in sorted(groups.items())},
            "by_family":{k:summary([r for r in rows if r["family"]==k]) for k in sorted({r["family"] for r in rows})},
            "timing":perf,
            "candidate_overhead_fraction":perf["candidate"]["median_ns"]/perf["r21"]["median_ns"]-1,
            "changed_examples":[r for r in rows if r["changed"]][:65]}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"h13.json").write_text(json.dumps(result,indent=2))
    print("H13_RESULT_START")
    print(json.dumps({k:v for k,v in result.items() if k!="changed_examples"},indent=2))
    print("H13_RESULT_END")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",required=True)
    p.add_argument("--out",default="research-results/h13")
    main(p.parse_args())
