from __future__ import annotations
import argparse, hashlib, json, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import chardet
from charset_normalizer import from_bytes
from codeccat import load_bundle
from codeccat.features import hmt768_array,baseline_from_array,full_b_extra_from_array
from codeccat.runtime import Runtime,_signals,_context,family
from h2_fused_candidate_mask_ab import collect,dedupe,normalize

CRAWLS=("CC-MAIN-2026-39","CC-MAIN-2026-34","CC-MAIN-2026-21","CC-MAIN-2026-17")
STAGES=("raw_baseline","baseline_downstream","baseline_with_rl","frozen_r21","full_b_always","full_b_raw")
def assess(rt,data):
    b=rt.bundle
    sig=_signals(data)
    array=hmt768_array(data)
    v=baseline_from_array(array)
    bc=_context(data,sig,b.baseline.route_models[sig.route],v)
    raw=bc.rank
    downstream=tuple(rt.baseline.rank_context(bc))
    rl=tuple(rt._apply_rl_specialist(bc,list(downstream)))
    margin=float(bc.sorted_scores[0]-bc.sorted_scores[1]) if len(bc.sorted_scores)>=2 else float("inf")
    gate=b.gate_thresholds.get(sig.route)
    escalate=rt.full_b is not None and gate is not None and sig.route in ("U","RH") and margin<=gate
    full=None;full_raw=None
    if rt.full_b is not None and sig.route in ("U","RH"):
        extra=full_b_extra_from_array(array,sig.route)
        fv=np.concatenate([v,extra])
        fc=_context(data,sig,b.full_b.route_models[sig.route],fv)
        full=tuple(rt.full_b.rank_context(fc))
        full_raw=tuple(fc.rank)
    frozen=full if escalate else rl
    return sig.route,{
        "raw_baseline":raw,"baseline_downstream":downstream,
        "baseline_with_rl":rl,"frozen_r21":frozen,
        "full_b_always":full if full is not None else rl,
        "full_b_raw":full_raw if full_raw is not None else raw,
    },escalate

def eval_rows(rows,rt):
    stage=defaultdict(Counter); route=defaultdict(lambda:defaultdict(Counter))
    group=defaultdict(lambda:defaultdict(Counter))
    deltas=defaultdict(Counter)
    disagreement=Counter()
    margins=defaultdict(Counter)
    for s in rows:
        data=s["data"]; truth=normalize(s["label"])
        r,preds,gate=assess(rt,data)
        comp={
            "chardet7":normalize(chardet.detect(data).get("encoding")),
            "charset_normalizer":None
        }
        result=from_bytes(data).best()
        comp["charset_normalizer"]=normalize(result.encoding) if result is not None else None
        outcomes={**{k:(v[0] if v else None) for k,v in preds.items()},**comp}
        for k,v in outcomes.items():
            correct=v==truth
            stage[k]["n"]+=1;stage[k]["hits"]+=correct
            route[r][k]["n"]+=1;route[r][k]["hits"]+=correct
            group[family(truth)][k]["n"]+=1;group[family(truth)][k]["hits"]+=correct
        for a,b in zip(STAGES,STAGES[1:]):
            if outcomes[a]!=outcomes[b]:
                deltas[a+"__to__"+b]["changed"]+=1
                deltas[a+"__to__"+b]["improved"]+=outcomes[a]!=truth and outcomes[b]==truth
                deltas[a+"__to__"+b]["harmed"]+=outcomes[a]==truth and outcomes[b]!=truth
        for k in ("chardet7","charset_normalizer"):
            if outcomes[k]==truth and outcomes["frozen_r21"]!=truth:
                disagreement[k+"_right_codeccat_wrong"]+=1
                disagreement[k+"_right_codeccat_wrong_route_"+r]+=1
                disagreement[k+"_right_codeccat_wrong_family_"+family(truth)]+=1
            if outcomes[k]!=truth and outcomes["frozen_r21"]==truth:
                disagreement["codeccat_right_"+k+"_wrong"]+=1
        if gate:
            margins["escalated"]["n"]+=1
        if outcomes["full_b_always"]!=outcomes["frozen_r21"]:
            margins["always_vs_frozen"]["changed"]+=1
            margins["always_vs_frozen"]["improved"]+=outcomes["full_b_always"]==truth and outcomes["frozen_r21"]!=truth
            margins["always_vs_frozen"]["harmed"]+=outcomes["frozen_r21"]==truth and outcomes["full_b_always"]!=truth
    def convert(x):
        return {k:dict(v) for k,v in x.items()}
    return {"stages":convert(stage),"by_route":{r:convert(v) for r,v in route.items()},
            "by_truth_family":{r:convert(v) for r,v in group.items()},
            "stage_transitions":convert(deltas),"competitor_disagreements":dict(disagreement),
            "full_b_gate":convert(margins)}

def timing(rows,rt):
    data=[s["data"] for s in rows]
    out={}
    for label,fn in (("frozen_r21",rt.rank),("full_b_always",lambda d:assess(rt,d)[1]["full_b_always"])):
        vals=[]
        for _ in range(7):
            st=time.perf_counter_ns()
            sink=sum(len(fn(d)) for d in data)
            vals.append((time.perf_counter_ns()-st)/max(1,len(data)))
        out[label]={"median_ns_per_sample":float(np.median(vals)),"min_ns_per_sample":min(vals),"max_ns_per_sample":max(vals),"sink":sink}
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--sha256",required=True);a=p.parse_args()
    if hashlib.sha256(Path(a.model).read_bytes()).hexdigest()!=a.sha256:raise ValueError("immutable model mismatch")
    rt=Runtime(load_bundle(Path(a.model)))
    pooled=[]; per={};meta={}
    for crawl in CRAWLS:
        rows,paths,stats=collect(crawl);rows=dedupe(rows)
        per[crawl]=eval_rows(rows,rt)
        pooled.extend(rows)
        meta[crawl]={"n":len(rows),"warc_paths_considered":len(paths),"stats":stats}
    pooled=dedupe(pooled)
    result={"study":"Accuracy Recovery Diagnostic — no tuning",
            "constraints":{"release_modified":False,"retraining":False,"threshold_search":False,"r22_used":False,"cc30_used":False},
            "metadata":meta,"per_crawl":per,"pooled":eval_rows(pooled,rt),
            "runtime":timing(pooled,rt)}
    print("DIAGNOSTIC_JSON_START")
    print(json.dumps(result,sort_keys=True,indent=2))
    print("DIAGNOSTIC_JSON_END")
if __name__=="__main__":main()
