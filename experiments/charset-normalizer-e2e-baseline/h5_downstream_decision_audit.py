from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from codeccat import load_bundle
from codeccat.runtime import Runtime,_signals,_context,_rule_rerank,replacement_rate,RATE_THRESHOLD
from codeccat.features import hmt768_array,baseline_from_array,full_b_extra_from_array
from h2_fused_candidate_mask_ab import collect,dedupe,normalize

CRAWLS=("CC-MAIN-2026-39","CC-MAIN-2026-34","CC-MAIN-2026-21","CC-MAIN-2026-17")
STAGES=("raw","rule","hybrid","triad_gate","utf8_sig_pair","utf8_gb_pair","replacement_guard","rl_specialist")

def trace(stack,ctx):
    raw=list(ctx.rank)
    rule=_rule_rerank(ctx)
    hybrid=stack._hybrid(ctx)
    gate=stack._gate(ctx,hybrid)
    sig=stack._pair(stack.stack.utf8_sig,("utf-8","utf-8-sig"),ctx,gate)
    gb=stack._pair(stack.stack.utf8_gb,("utf-8","gb18030"),ctx,sig)
    guard=gb
    if (gb and sig and gb[0]!=sig[0] and sig[0]=="gb18030" and gb[0]=="utf-8" and replacement_rate(ctx.data)>RATE_THRESHOLD):
        guard=sig
    actual=stack.rank_context(ctx)
    assert guard==actual,("trace not equivalent",ctx.route,guard[:1],actual[:1])
    return {"raw":raw,"rule":rule,"hybrid":hybrid,"triad_gate":gate,
            "utf8_sig_pair":sig,"utf8_gb_pair":gb,"replacement_guard":guard}

def observe(rows,rt):
    stages=defaultdict(Counter)
    by_route=defaultdict(lambda:defaultdict(Counter))
    by_family=defaultdict(lambda:defaultdict(Counter))
    transitions=defaultdict(Counter)
    contrast=Counter()
    for s in rows:
        data=s["data"];truth=normalize(s["label"]); sig=_signals(data)
        a=hmt768_array(data);v=baseline_from_array(a)
        bc=_context(data,sig,rt.bundle.baseline.route_models[sig.route],v)
        btrace=trace(rt.baseline,bc)
        btrace["rl_specialist"]=rt._apply_rl_specialist(bc,btrace["replacement_guard"])
        margin=float(bc.sorted_scores[0]-bc.sorted_scores[1]) if len(bc.sorted_scores)>1 else float("inf")
        threshold=rt.bundle.gate_thresholds.get(sig.route)
        escalated=rt.full_b is not None and threshold is not None and sig.route in ("U","RH") and margin<=threshold
        if rt.full_b is not None and sig.route in ("U","RH"):
            e=full_b_extra_from_array(a,sig.route);fv=np.concatenate((v,e))
            fc=_context(data,sig,rt.bundle.full_b.route_models[sig.route],fv)
            ftrace=trace(rt.full_b,fc)
        else:ftrace=None
        for stack_name,t in (("baseline",btrace),("full_b",ftrace)):
            if t is None:continue
            prev=None
            for k,rank in t.items():
                name=stack_name+"."+k; pred=rank[0] if rank else None; hit=pred==truth
                stages[name]["n"]+=1;stages[name]["hits"]+=int(hit)
                by_route[sig.route][name]["n"]+=1;by_route[sig.route][name]["hits"]+=int(hit)
                by_family[truth][name]["n"]+=1;by_family[truth][name]["hits"]+=int(hit)
                if prev:
                    pk,pr=prev; old=pr[0] if pr else None
                    c=transitions[stack_name+"."+pk+"->"+k]
                    c["n"]+=1;c["changed"]+=int(old!=pred)
                    c["improved"]+=int(old!=truth and hit)
                    c["harmed"]+=int(old==truth and not hit)
                    if old!=pred:
                        c["route_"+sig.route+"_changed"]+=1
                        c["truth_"+truth+"_changed"]+=1
                        c["old_"+str(old)+"_to_"+str(pred)]+=1
                prev=(k,rank)
        b=btrace["rl_specialist"][0]
        f=ftrace["replacement_guard"][0] if ftrace else b
        final=f if escalated else b
        contrast["n"]+=1
        contrast["frozen_hits"]+=int(final==truth)
        contrast["baseline_hits"]+=int(b==truth)
        contrast["full_b_always_hits"]+=int(f==truth)
        contrast["escalation_n"]+=int(escalated)
        if final!=b:
            contrast["frozen_changed_baseline"]+=1
            contrast["frozen_beneficial"]+=int(final==truth and b!=truth)
            contrast["frozen_harmful"]+=int(final!=truth and b==truth)
        if f!=final:
            contrast["always_changed_frozen"]+=1
            contrast["always_beneficial"]+=int(f==truth and final!=truth)
            contrast["always_harmful"]+=int(f!=truth and final==truth)
    def conv(v):return {k:dict(c) for k,c in v.items()}
    return {"stages":conv(stages),"transitions":conv(transitions),
            "by_route":{r:conv(z) for r,z in by_route.items()},
            "by_truth_label":{r:conv(z) for r,z in by_family.items()},
            "contrast":dict(contrast)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--sha256",required=True);args=p.parse_args()
    assert hashlib.sha256(Path(args.model).read_bytes()).hexdigest()==args.sha256
    rt=Runtime(load_bundle(Path(args.model)));pooled=[];per={};meta={}
    # Smoke-check every production route before fetching remote WARC data.
    smoke=[
        {"data":b"hello", "label":"utf-8"},
        {"data":b"hello\\x80", "label":"cp1252"},
        {"data":bytes([0x80,0x81])*300, "label":"cp1251"},
        {"data":b"\\x00\\xff"*300, "label":"utf-16-le"},
    ]
    # Literal bytes are constructed here; this diagnostic preflight is not scored.
    smoke[1]["data"]=b"A"*512+bytes([0x80])
    smoke[3]["data"]=bytes([0,255])*300
    assert [_signals(s["data"]).route for s in smoke]==["U","RL","RH","N"]
    smoke_result=observe(smoke,rt)
    assert smoke_result["contrast"]["n"]==4, "route smoke preflight failed"
    print("H5_ROUTE_PREFLIGHT_PASS", flush=True)
    for crawl in CRAWLS:
        rows,paths,stats=collect(crawl);rows=dedupe(rows);pooled.extend(rows)
        per[crawl]=observe(rows,rt);meta[crawl]={"n":len(rows),"paths":len(paths),"stats":stats}
    pooled=dedupe(pooled)
    print("H5_JSON_START")
    print(json.dumps({"study":"H5 frozen downstream stage audit","model_modified":False,
       "parameters_tuned":False,"r22_used":False,"metadata":meta,
       "per_crawl":per,"pooled":observe(pooled,rt)},sort_keys=True))
    print("H5_JSON_END")
if __name__=="__main__":main()
