from __future__ import annotations

import argparse
import json
import math
import time

import joblib

import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_p1_cached_inference_context_ab as p1
import charset_p2_multihop_kernel_tournament as p2
import charset_p3_allocation_lazy_tournament as p3
import charset_p5_branch_elision_tournament as p5

TIMING_REPEATS=17
# Cache-hit verification trigger.

def in_top3(rank,label):
    return rank[0]==label or rank[1]==label or rank[2]==label

class P6Downstream(p5.P5Downstream):
    def __init__(self,*args,no_rank_copy=False,direct_membership=False,**kwargs):
        super().__init__(*args,skip_cal_on_rule=True,precompute_cal_static=True,**kwargs)
        self.no_rank_copy=no_rank_copy
        self.direct_membership=direct_membership

    def hybrid(self,ctx):
        rule=p1.cached_rule_rerank(ctx)
        if self.no_rank_copy:
            if rule and ctx.rank and rule[0]!=ctx.rank[0]:
                return rule
            return self.choose_cal(ctx)
        return super().hybrid(ctx)

    def gate(self,ctx,hybrid):
        if not self.direct_membership:
            return super().gate(ctx,hybrid)
        if self.triad is None or not hybrid or hybrid[0] not in tri.TRIAD:
            return hybrid
        r=ctx.rank
        count=int(r[0] in tri.TRIAD)+int(r[1] in tri.TRIAD)+int(r[2] in tri.TRIAD)
        if count<2:
            return hybrid
        classes,logits=self.triad_logits(ctx)
        a=float(logits[0]); b=float(logits[1]); c=float(logits[2])
        if a>=b and a>=c:
            idx=0; m=a; o1=b; o2=c
        elif b>=c:
            idx=1; m=b; o1=a; o2=c
        else:
            idx=2; m=c; o1=a; o2=b
        if (math.exp(o1-m)+math.exp(o2-m))>1.0:
            return hybrid
        pred=classes[idx]
        if not in_top3(r,pred):
            return hybrid
        out=list(hybrid)
        if pred in out:
            out.remove(pred); out.insert(0,pred)
        return out

    def pair(self,fused,pair,ctx,baseline):
        if not self.direct_membership:
            return super().pair(fused,pair,ctx,baseline)
        if fused is None or not baseline or baseline[0] not in pair:
            return baseline
        r=ctx.rank
        if not in_top3(r,pair[0]) or not in_top3(r,pair[1]):
            return baseline
        idxs=self.sig_idx[ctx.route] if pair==p1.SIG_PAIR else self.gb_idx[ctx.route]
        classes,z=self.pair_logit_indexed(fused,idxs,ctx)
        pred=classes[1] if z>0.0 else classes[0]
        if not in_top3(r,pred):
            return baseline
        out=list(baseline)
        if pred in out:
            out.remove(pred); out.insert(0,pred)
        return out

CANDIDATES={
    "p5_baseline":dict(no_rank_copy=False,direct_membership=False),
    "no_rank_copy":dict(no_rank_copy=True,direct_membership=False),
    "direct_membership":dict(no_rank_copy=False,direct_membership=True),
    "combined":dict(no_rank_copy=True,direct_membership=True),
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",required=True)
    args=ap.parse_args()
    state=joblib.load(args.state)
    assert state["schema_version"]==1
    assert sum(len(f["rows"]) for f in state["folds"])==418

    classes=state["classes"]; families=state["families"]
    pooled={n:{"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}
    folds=[]

    for fd in state["folds"]:
        models=fd["models"]; cal=fd["cal"]; triad_cal=fd["triad_cal"]
        sig_cal=fd["sig_cal"]; gb_cal=fd["gb_cal"]; rows=fd["rows"]
        fused={r:p2.fuse_scaler_linear(mt) for r,mt in models.items()}
        dsmap={
            name:P6Downstream(
                cal,triad_cal,sig_cal,gb_cal,classes,families,models,True,
                no_rank_copy=cfg["no_rank_copy"],
                direct_membership=cfg["direct_membership"],
            )
            for name,cfg in CANDIDATES.items()
        }

        refs=[]
        for s in rows:
            ctx=p3.build_ctx(models[s["route"]],s,fused[s["route"]],True)
            refs.append(dsmap["p5_baseline"].run(ctx))

        fs={n:{"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}
        for s,ref in zip(rows,refs):
            mt=models[s["route"]]
            for name in CANDIDATES:
                ctx=p3.build_ctx(mt,s,fused[s["route"]],True)
                out=dsmap[name].run(ctx)
                st=fs[name]
                st["n"]+=1
                st["hits"]+=int(out and out[0]==s["label"])
                st["mismatch"]+=int(list(out)!=list(ref))

        for name in CANDIDATES:
            if fs[name]["mismatch"]:
                continue
            sink=0
            t0=time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    ctx=p3.build_ctx(models[s["route"]],s,fused[s["route"]],True)
                    out=dsmap[name].run(ctx)
                    sink+=len(out)
            dt=time.perf_counter_ns()-t0
            fs[name]["elapsed"]=dt
            fs[name]["calls"]=len(rows)*TIMING_REPEATS
            fs[name]["sink"]=sink

        fr={"fold":fd["fold"],"n":len(rows),"candidates":{}}
        for name,st in fs.items():
            fr["candidates"][name]={
                "top1":st["hits"]/max(1,st["n"]),
                "mismatch_n":st["mismatch"],
                "ns_per_call":st["elapsed"]/st["calls"] if st["calls"] else None,
            }
            for k in ("n","hits","mismatch","elapsed","calls"):
                pooled[name][k]+=st[k]
        folds.append(fr)

    out={}
    for name,st in pooled.items():
        ns=st["elapsed"]/st["calls"] if st["calls"] else None
        out[name]={
            "n":st["n"],"hits":st["hits"],
            "top1":st["hits"]/max(1,st["n"]),
            "mismatch_n":st["mismatch"],
            "ns_per_call":ns,
        }
    base_ns=out["p5_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p5"]=base_ns/row["ns_per_call"]
            row["runtime_reduction_vs_p5"]=1-row["ns_per_call"]/base_ns

    eligible=[(r["ns_per_call"],n) for n,r in out.items()
              if r["mismatch_n"]==0 and r["hits"]==365 and r["ns_per_call"] is not None]
    winner=min(eligible)[1] if eligible else None

    print(json.dumps({
        "phase":"p6_cached_state_residual_tournament",
        "canonical_target":{"n":418,"hits":365,"top1":365/418},
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "cached_training_seconds":state.get("training_seconds"),
        "timing_repeats":TIMING_REPEATS,
        "pooled":out,"winner":winner,"folds":folds,
        "decision_rule":"Preserve exact P5 ranking and 365/418; choose fastest same-run candidate."
    },indent=2,sort_keys=True))

if __name__=="__main__":
    main()
