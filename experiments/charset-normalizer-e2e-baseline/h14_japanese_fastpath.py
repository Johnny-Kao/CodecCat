"""H14 terminal fast path; exact H13 semantics, single Shift-JIS decode only."""
import argparse,hashlib,json,statistics,time
from collections import defaultdict
from pathlib import Path
from codeccat import load_bundle
from codeccat.runtime import Runtime,_signals
from h2_fused_candidate_mask_ab import collect,normalize
from h13_japanese_evidence_ab import CRAWLS,SHA,candidate as h13

def fast(data,baseline):
    if baseline=="shift-jis" or _signals(data).route!="RH":return baseline,False
    raw=data[:4096]
    for trim in range(min(4,len(raw))+1):
        try:
            decoded=raw[:len(raw)-trim].decode("shift_jis","strict")
            break
        except UnicodeDecodeError:
            continue
    else:return baseline,False
    kana=0
    for char in decoded:
        if ("\u3040"<=char<="\u309f") or ("\u30a0"<=char<="\u30ff"):
            kana+=1
            if kana>=2:return "shift-jis",True
    return baseline,False

def main(a):
    assert hashlib.sha256(Path(a.model).read_bytes()).hexdigest()==SHA
    rt=Runtime(load_bundle(a.model))
    rows=[];seen=set()
    for crawl in CRAWLS:
        data_rows,_,_=collect(crawl)
        for item in data_rows:
            data=item["data"];sha=hashlib.sha256(data).hexdigest()
            if sha in seen:continue
            seen.add(sha)
            base=rt.rank(data)[0]
            old,_=h13(data,base)
            new,_=fast(data,base)
            if old!=new:raise AssertionError(f"H13 semantic mismatch {sha} {old} {new}")
            rows.append((crawl,data,normalize(item["label"]),base,new))
        print("H14_CRAWL",crawl,len(data_rows),flush=True)
    by=defaultdict(list)
    for crawl,data,truth,base,new in rows:
        by[crawl].append((truth,base,new))
    def metrics(rows):
        return {"n":len(rows),"r21":sum(y==b for y,b,c in rows),"h14":sum(y==c for y,b,c in rows),
        "helped":sum(y!=b and y==c for y,b,c in rows),"harmed":sum(y==b and y!=c for y,b,c in rows)}
    timing={}
    for name,fn in (("r21",None),("h13",h13),("h14",fast)):
        times=[]
        for i in range(5):
            batch=rows if i%2==0 else list(reversed(rows))
            t=time.perf_counter_ns()
            for _,data,_,_,_ in batch:
                pred=rt.rank(data)[0]
                if fn:fn(data,pred)
            times.append((time.perf_counter_ns()-t)/len(rows))
        timing[name]={"median_ns":statistics.median(times)}
    total=metrics([(truth,base,new) for _,_,truth,base,new in rows])
    result={"total":total,"by_crawl":{k:metrics(v) for k,v in sorted(by.items())},"timing":timing,
            "speedup_vs_h13":timing["h13"]["median_ns"]/timing["h14"]["median_ns"],
            "overhead_vs_r21":timing["h14"]["median_ns"]/timing["r21"]["median_ns"],
            "exact_match_h13":True,
            "limitations":"Same previously exposed crawl sample selection, no independent holdout; label declarations unverified"}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2))
    print("H14_RESULT_START");print(json.dumps(result,indent=2));print("H14_RESULT_END")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--out",default="research-results/h14")
    main(p.parse_args())
