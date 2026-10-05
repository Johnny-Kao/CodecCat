from __future__ import annotations

import hashlib
import json
import time

import numpy as np

import charset_external_scorer_learning_curve as lc
import charset_external_scorer_domain_shift_ab as base
import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_canonical_guarded_final_validation as canon
import charset_p1_cached_inference_context_ab as p1
import charset_p2_multihop_kernel_tournament as p2
import charset_p3_allocation_lazy_tournament as p3

OUTER_FOLDS=4
TRAIN_N=300
TIMING_REPEATS=11

class IndexedDirectDownstream(p3.DirectDownstream):
    def __init__(self,cal,triad_cal,sig_cal,gb_cal,classes,families,route_models,branch_precompute=False):
        super().__init__(cal,triad_cal,sig_cal,gb_cal,classes,families,True)
        self.branch_precompute=branch_precompute
        self.model_class_idx={
            route:{c:i for i,c in enumerate(mt[1].classes_)}
            for route,mt in route_models.items()
        }
        self.triad_idx={
            route:tuple(self.model_class_idx[route].get(c,-1) for c in tri.TRIAD)
            for route in route_models
        }
        self.sig_idx={
            route:tuple(self.model_class_idx[route].get(c,-1) for c in p1.SIG_PAIR)
            for route in route_models
        }
        self.gb_idx={
            route:tuple(self.model_class_idx[route].get(c,-1) for c in p1.GB_PAIR)
            for route in route_models
        }
        if branch_precompute:
            self.route_weight_idx={
                r:(7+self.route_idx[r],10+self.route_idx[r],13+self.route_idx[r])
                for r in self.route_idx
            }

    @staticmethod
    def _raw(ctx,idx):
        return float(ctx.raw_scores[idx]) if idx>=0 else -20.0

    def triad_logits(self,ctx):
        classes,w,b=self.triad
        i0,i1,i2=self.triad_idx[ctx.route]
        v0=self._raw(ctx,i0); v1=self._raw(ctx,i1); v2=self._raw(ctx,i2)
        second=max(min(v0,v1),min(max(v0,v1),v2))
        xs=(
            v0,v1,v2,
            v0-v1,v0-v2,v1-v2,max(v0,v1,v2)-second,
            float(ctx.strict_utf8),float(ctx.has_utf8_bom),float(ctx.ascii_only),
            ctx.high_byte_ratio,ctx.nul_ratio,float(ctx.sample_len_4k)
        )
        out=np.array(b,dtype=np.float64,copy=True)
        for j,x in enumerate(xs): out+=w[:,j]*x
        ridx=self.route_weight_idx[ctx.route][2] if self.branch_precompute else 13+self.route_idx[ctx.route]
        out+=w[:,ridx]
        return classes,out

    def pair_logit_indexed(self,fused,idxs,ctx):
        classes,w,b=fused; v=w[0]
        sa=self._raw(ctx,idxs[0]); sb=self._raw(ctx,idxs[1]); d=sa-sb
        xs=(sa,sb,d,abs(d),float(ctx.strict_utf8),float(ctx.has_utf8_bom),float(ctx.ascii_only),
            ctx.high_byte_ratio,ctx.nul_ratio,float(ctx.sample_len_4k))
        z=float(b[0])
        for j,x in enumerate(xs): z+=v[j]*x
        ridx=self.route_weight_idx[ctx.route][1] if self.branch_precompute else 10+self.route_idx[ctx.route]
        z+=v[ridx]
        return classes,z

    def pair(self,fused,pair,ctx,baseline):
        if fused is None or not baseline or baseline[0] not in pair: return baseline
        if not all(p in ctx.rank[:3] for p in pair): return baseline
        idxs=self.sig_idx[ctx.route] if pair==p1.SIG_PAIR else self.gb_idx[ctx.route]
        classes,z=self.pair_logit_indexed(fused,idxs,ctx)
        pred=classes[1] if z>0.0 else classes[0]
        if pred not in ctx.rank[:3]: return baseline
        out=list(baseline)
        if pred in out: out.remove(pred); out.insert(0,pred)
        return out

    def cal_logit(self,ctx,cand,i,score,top,second):
        _,w,b=self.cal
        v=w[0]
        z=float(b[0])
        z+=v[0]*score+v[1]*(score-top)+v[2]*(score-second)+v[3]*i
        z+=v[4]*float(ctx.has_utf8_bom)+v[5]*float(ctx.strict_utf8)+v[6]*float(ctx.ascii_only)
        ridx=self.route_weight_idx[ctx.route][0] if self.branch_precompute else 7+self.route_idx[ctx.route]
        z+=v[ridx]
        ci=self.class_idx.get(cand)
        if ci is not None: z+=v[11+ci]
        fi=self.fam_idx.get(calmod.fam(cand))
        if fi is not None: z+=v[11+len(self.classes)+fi]
        return z

CANDIDATES=("p2_baseline","direct_scalar_lazy","direct_scalar_lazy_indexed","indexed_branch_precompute")

def main():
    external,_,stats=base.collect_external()
    legacy_X,legacy_y,legacy_b=base.load_legacy_training()
    classes,families=calmod.build_vocab(legacy_y)
    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}
    pooled={n:{"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}
    folds=[]

    for fold in range(OUTER_FOLDS):
        train=[s for s in external if fold_assign[s["warc_path"]]!=fold]
        test=[s for s in external if fold_assign[s["warc_path"]]==fold]
        ext=lc.deterministic_nested_subset(train,TRAIN_N)
        models=lc.fit_route_models(ext,legacy_X,legacy_y,legacy_b)
        cal=calmod.crossfit_calibration_rows(train,legacy_X,legacy_y,legacy_b,classes,families)
        triad_cal=tri.crossfit_triad_rows(train,legacy_X,legacy_y,legacy_b,classes,families)
        sig_cal=canon.fit_one(canon.SIG_PAIR,train,legacy_X,legacy_y,legacy_b,classes,families)
        gb_cal=canon.fit_one(canon.GB_PAIR,train,legacy_X,legacy_y,legacy_b,classes,families)
        rows=[s for s in test if models.get(s["route"]) is not None and s["label"] in models[s["route"]][1].classes_]
        fused={r:p2.fuse_scaler_linear(mt) for r,mt in models.items()}
        p2ds=p2.FastDownstream(cal,triad_cal,sig_cal,gb_cal,classes,families)
        direct=p3.DirectDownstream(cal,triad_cal,sig_cal,gb_cal,classes,families,True)
        indexed=IndexedDirectDownstream(cal,triad_cal,sig_cal,gb_cal,classes,families,models,False)
        indexed_pre=IndexedDirectDownstream(cal,triad_cal,sig_cal,gb_cal,classes,families,models,True)

        refs=[]
        for s in rows:
            ctx=p2.build_ctx(models[s["route"]],s,"combined","fused",fused[s["route"]],True)
            refs.append(p2ds.run(ctx))

        fs={n:{"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}
        for s,ref in zip(rows,refs):
            mt=models[s["route"]]
            ctx2=p2.build_ctx(mt,s,"combined","fused",fused[s["route"]],True)
            ctx=p3.build_ctx(mt,s,fused[s["route"]],True)
            outs={
                "p2_baseline":p2ds.run(ctx2),
                "direct_scalar_lazy":direct.run(ctx),
                "direct_scalar_lazy_indexed":indexed.run(ctx),
                "indexed_branch_precompute":indexed_pre.run(ctx),
            }
            for name,out in outs.items():
                st=fs[name]; st["n"]+=1; st["hits"]+=int(out and out[0]==s["label"]); st["mismatch"]+=int(list(out)!=list(ref))

        for name in CANDIDATES:
            if fs[name]["mismatch"]: continue
            t0=time.perf_counter_ns(); sink=0
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    mt=models[s["route"]]
                    if name=="p2_baseline":
                        ctx=p2.build_ctx(mt,s,"combined","fused",fused[s["route"]],True); out=p2ds.run(ctx)
                    else:
                        ctx=p3.build_ctx(mt,s,fused[s["route"]],True)
                        if name=="direct_scalar_lazy": out=direct.run(ctx)
                        elif name=="direct_scalar_lazy_indexed": out=indexed.run(ctx)
                        else: out=indexed_pre.run(ctx)
                    sink+=len(out)
            dt=time.perf_counter_ns()-t0
            fs[name]["elapsed"]=dt; fs[name]["calls"]=len(rows)*TIMING_REPEATS; fs[name]["sink"]=sink

        fr={"fold":fold,"n":len(rows),"candidates":{}}
        for name,st in fs.items():
            fr["candidates"][name]={
                "top1":st["hits"]/max(1,st["n"]),
                "mismatch_n":st["mismatch"],
                "ns_per_call":st["elapsed"]/st["calls"] if st["calls"] else None
            }
            for k in ("n","hits","mismatch","elapsed","calls"): pooled[name][k]+=st[k]
        folds.append(fr)

    out={}
    for name,st in pooled.items():
        ns=st["elapsed"]/st["calls"] if st["calls"] else None
        out[name]={"n":st["n"],"hits":st["hits"],"top1":st["hits"]/max(1,st["n"]),"mismatch_n":st["mismatch"],"ns_per_call":ns}
    p2_ns=out["p2_baseline"]["ns_per_call"]; p3_ns=out["direct_scalar_lazy"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p2"]=p2_ns/row["ns_per_call"]
            row["speedup_vs_p3_direct"]=p3_ns/row["ns_per_call"]
            row["runtime_reduction_vs_p3_direct"]=1-row["ns_per_call"]/p3_ns
    eligible=[(r["ns_per_call"],n) for n,r in out.items() if r["mismatch_n"]==0 and r["hits"]==365 and r["ns_per_call"] is not None]
    winner=min(eligible)[1] if eligible else None
    print(json.dumps({
        "phase":"p3_combined_winner_tournament",
        "canonical_target":{"n":418,"hits":365,"top1":365/418},
        "timing_repeats":TIMING_REPEATS,
        "pooled":out,"winner":winner,"folds":folds,
        "collection_stats":dict(stats),
        "decision_rule":"Preserve 365/418 and exact P2 ranking; choose fastest pooled candidate."
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
