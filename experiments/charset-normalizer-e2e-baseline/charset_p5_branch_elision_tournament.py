from __future__ import annotations

import hashlib, json, time
import numpy as np

import charset_external_scorer_learning_curve as lc
import charset_external_scorer_domain_shift_ab as base
import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_canonical_guarded_final_validation as canon
import charset_p1_cached_inference_context_ab as p1
import charset_p2_multihop_kernel_tournament as p2
import charset_p3_allocation_lazy_tournament as p3
import charset_p3_combined_winner_tournament as p3c
import charset_p4_residual_python_tournament as p4

OUTER_FOLDS=4
TRAIN_N=300
TIMING_REPEATS=15
# Retry-safe external corpus fetch is provided by the shared collector.\n# Retrigger after transient-range retry hardening.

class P5Downstream(p4.P4Downstream):
    def __init__(self,*args,skip_cal_on_rule=False,precompute_cal_static=False,**kwargs):
        super().__init__(*args,manual_smallops=True,no_list_remove=False,**kwargs)
        self.skip_cal_on_rule=skip_cal_on_rule
        self.precompute_cal_static=precompute_cal_static
        self.cal_static={}
        if precompute_cal_static and self.cal is not None:
            _,w,_=self.cal
            v=w[0]
            for route in calmod.ROUTES:
                rterm=v[7+self.route_idx[route]]
                for cand in self.classes:
                    z=float(rterm)
                    ci=self.class_idx.get(cand)
                    if ci is not None:
                        z+=v[11+ci]
                    fi=self.fam_idx.get(calmod.fam(cand))
                    if fi is not None:
                        z+=v[11+len(self.classes)+fi]
                    self.cal_static[(route,cand)]=float(z)

    def cal_logit(self,ctx,cand,i,score,top,second):
        if not self.precompute_cal_static:
            return super().cal_logit(ctx,cand,i,score,top,second)
        _,w,b=self.cal
        v=w[0]
        z=float(b[0])
        z+=v[0]*score+v[1]*(score-top)+v[2]*(score-second)+v[3]*i
        z+=v[4]*float(ctx.has_utf8_bom)+v[5]*float(ctx.strict_utf8)+v[6]*float(ctx.ascii_only)
        z+=self.cal_static[(ctx.route,cand)]
        return z

    def hybrid(self,ctx):
        rank=list(ctx.rank)
        rule=self.rule_rerank(ctx)
        if self.skip_cal_on_rule and rule and rank and rule[0]!=rank[0]:
            return rule
        cal=self.choose_cal(ctx)
        return rule if (rule and rank and rule[0]!=rank[0]) else cal

CANDIDATES={
    "p4_baseline":dict(skip=False,static=False),
    "skip_cal_on_rule":dict(skip=True,static=False),
    "precompute_cal_static":dict(skip=False,static=True),
    "skip_plus_static":dict(skip=True,static=True),
}

def main():
    external,_,stats=base.collect_external()
    legacy_X,legacy_y,legacy_b=base.load_legacy_training()
    classes,families=calmod.build_vocab(legacy_y)
    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}

    pooled={n:{"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}
    folds=[]
    training_ns=0

    for fold in range(OUTER_FOLDS):
        train=[s for s in external if fold_assign[s["warc_path"]]!=fold]
        test=[s for s in external if fold_assign[s["warc_path"]]==fold]
        ext=lc.deterministic_nested_subset(train,TRAIN_N)

        tfit=time.perf_counter_ns()
        models=lc.fit_route_models(ext,legacy_X,legacy_y,legacy_b)
        cal=calmod.crossfit_calibration_rows(train,legacy_X,legacy_y,legacy_b,classes,families)
        triad_cal=tri.crossfit_triad_rows(train,legacy_X,legacy_y,legacy_b,classes,families)
        sig_cal=canon.fit_one(canon.SIG_PAIR,train,legacy_X,legacy_y,legacy_b,classes,families)
        gb_cal=canon.fit_one(canon.GB_PAIR,train,legacy_X,legacy_y,legacy_b,classes,families)
        training_ns+=time.perf_counter_ns()-tfit

        rows=[s for s in test if models.get(s["route"]) is not None and s["label"] in models[s["route"]][1].classes_]
        fused={r:p2.fuse_scaler_linear(mt) for r,mt in models.items()}

        dsmap={
            name:P5Downstream(
                cal,triad_cal,sig_cal,gb_cal,classes,families,models,True,
                skip_cal_on_rule=cfg["skip"],precompute_cal_static=cfg["static"]
            )
            for name,cfg in CANDIDATES.items()
        }

        refs=[]
        for s in rows:
            ctx=p3.build_ctx(models[s["route"]],s,fused[s["route"]],True)
            refs.append(dsmap["p4_baseline"].run(ctx))

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

        fr={"fold":fold,"n":len(rows),"candidates":{}}
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

    base_ns=out["p4_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p4"]=base_ns/row["ns_per_call"]
            row["runtime_reduction_vs_p4"]=1-row["ns_per_call"]/base_ns

    eligible=[(r["ns_per_call"],n) for n,r in out.items()
              if r["mismatch_n"]==0 and r["hits"]==365 and r["ns_per_call"] is not None]
    winner=min(eligible)[1] if eligible else None

    print(json.dumps({
        "phase":"p5_branch_elision_multihop_tournament",
        "canonical_target":{"n":418,"hits":365,"top1":365/418},
        "timing_repeats":TIMING_REPEATS,
        "training_seconds":training_ns/1e9,
        "pooled":out,
        "winner":winner,
        "folds":folds,
        "collection_stats":dict(stats),
        "decision_rule":"Preserve exact P4 ranking and 365/418; choose fastest same-run pooled candidate."
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
