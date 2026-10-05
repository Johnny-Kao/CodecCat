from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass

import numpy as np

import charset_external_scorer_learning_curve as lc
import charset_external_scorer_domain_shift_ab as base
import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_canonical_guarded_final_validation as canon
import charset_minimal_reranker_ab as rr
import charset_p1_cached_inference_context_ab as p1
import charset_p2_multihop_kernel_tournament as p2

OUTER_FOLDS=4
TRAIN_N=300
TIMING_REPEATS=9

@dataclass(frozen=True)
class Ctx:
    data: bytes
    route: str
    rank: tuple[str,...]
    sorted_scores: np.ndarray
    raw_scores: np.ndarray
    classes: tuple[str,...]
    has_utf8_bom: bool
    strict_utf8: bool
    ascii_only: bool
    high_byte_ratio: float
    nul_ratio: float
    sample_len_4k: int

def build_ctx(model_tuple,s,fused_model,with_map=False):
    x=p2.features_combined(s["data"])
    raw=p2.fused_raw(fused_model,x)
    _,model=model_tuple
    order=np.argsort(raw)[::-1]
    sample=s["data"][:4096]
    arr=np.frombuffer(sample,dtype=np.uint8)
    strict=rr.strict_utf8(s["data"])
    bom=rr.has_utf8_bom(s["data"])
    if len(arr):
        high_count=int(np.count_nonzero(arr>=128))
        nul_count=int(np.count_nonzero(arr==0))
        high=high_count/len(arr); nul=nul_count/len(arr); ascii_flag=high_count==0
    else:
        high=nul=0.0; ascii_flag=True
    return Ctx(
        data=s["data"], route=s["route"],
        rank=tuple(model.classes_[order]),
        sorted_scores=np.asarray(raw)[order],
        raw_scores=np.asarray(raw),
        classes=tuple(model.classes_),
        has_utf8_bom=bom, strict_utf8=strict, ascii_only=ascii_flag,
        high_byte_ratio=high, nul_ratio=nul, sample_len_4k=len(arr)
    )

def score_of(ctx,label):
    try:
        return float(ctx.raw_scores[ctx.classes.index(label)])
    except ValueError:
        return -20.0

class DirectDownstream:
    def __init__(self,cal,triad_cal,sig_cal,gb_cal,classes,families,lazy_rate=True):
        self.cal=p2.fuse_estimator(cal)
        self.triad=p2.fuse_estimator(triad_cal)
        self.sig=p2.fuse_estimator(sig_cal)
        self.gb=p2.fuse_estimator(gb_cal)
        self.class_idx={c:i for i,c in enumerate(classes)}
        self.fam_idx={c:i for i,c in enumerate(families)}
        self.route_idx={r:i for i,r in enumerate(calmod.ROUTES)}
        self.classes=classes
        self.families=families
        self.lazy_rate=lazy_rate

    def cal_logit(self,ctx,cand,i,score,top,second):
        _,w,b=self.cal
        v=w[0]
        z=float(b[0])
        z+=v[0]*score+v[1]*(score-top)+v[2]*(score-second)+v[3]*i
        z+=v[4]*float(ctx.has_utf8_bom)+v[5]*float(ctx.strict_utf8)+v[6]*float(ctx.ascii_only)
        z+=v[7+self.route_idx[ctx.route]]
        ci=self.class_idx.get(cand)
        if ci is not None: z+=v[11+ci]
        fi=self.fam_idx.get(calmod.fam(cand))
        if fi is not None: z+=v[11+len(self.classes)+fi]
        return z

    def choose_cal(self,ctx):
        if self.cal is None or len(ctx.rank)<2: return list(ctx.rank)
        top=float(ctx.sorted_scores[0]); second=float(ctx.sorted_scores[1])
        cand=list(ctx.rank[:calmod.TOP_CANDIDATES])
        logits=[self.cal_logit(ctx,c,i,float(ctx.sorted_scores[i]),top,second) for i,c in enumerate(cand)]
        winner=int(np.argmax(logits))
        out=list(ctx.rank)
        if winner:
            chosen=out.pop(winner); out.insert(0,chosen)
        return out

    def hybrid(self,ctx):
        rank=list(ctx.rank)
        rule=p1.cached_rule_rerank(ctx)
        cal=self.choose_cal(ctx)
        return rule if (rule and rank and rule[0]!=rank[0]) else cal

    def triad_logits(self,ctx):
        classes,w,b=self.triad
        vals=[score_of(ctx,c) for c in tri.TRIAD]
        margins=[vals[0]-vals[1],vals[0]-vals[2],vals[1]-vals[2],max(vals)-sorted(vals)[-2]]
        xs=vals+margins+[
            float(ctx.strict_utf8),float(ctx.has_utf8_bom),float(ctx.ascii_only),
            ctx.high_byte_ratio,ctx.nul_ratio,float(ctx.sample_len_4k)
        ]
        out=np.array(b,dtype=np.float64,copy=True)
        for j,x in enumerate(xs): out+=w[:,j]*x
        out+=w[:,13+self.route_idx[ctx.route]]
        return classes,out

    def gate(self,ctx,hybrid):
        if self.triad is None or not hybrid or hybrid[0] not in tri.TRIAD: return hybrid
        if sum(c in tri.TRIAD for c in ctx.rank[:3])<2: return hybrid
        classes,logits=self.triad_logits(ctx)
        idx=int(np.argmax(logits))
        shifted=logits-np.max(logits); ex=np.exp(shifted)
        pmax=float(ex[idx]/ex.sum()); pred=classes[idx]
        if pmax<0.50 or pred not in ctx.rank[:3]: return hybrid
        out=list(hybrid)
        if pred in out: out.remove(pred); out.insert(0,pred)
        return out

    def pair_logit(self,fused,pair,ctx):
        classes,w,b=fused; v=w[0]
        sa=score_of(ctx,pair[0]); sb=score_of(ctx,pair[1]); d=sa-sb
        xs=[sa,sb,d,abs(d),float(ctx.strict_utf8),float(ctx.has_utf8_bom),float(ctx.ascii_only),
            ctx.high_byte_ratio,ctx.nul_ratio,float(ctx.sample_len_4k)]
        z=float(b[0])
        for j,x in enumerate(xs): z+=v[j]*x
        z+=v[10+self.route_idx[ctx.route]]
        return classes,z

    def pair(self,fused,pair,ctx,baseline):
        if fused is None or not baseline or baseline[0] not in pair: return baseline
        if not all(p in ctx.rank[:3] for p in pair): return baseline
        classes,z=self.pair_logit(fused,pair,ctx)
        pred=classes[1] if z>0.0 else classes[0]
        if pred not in ctx.rank[:3]: return baseline
        out=list(baseline)
        if pred in out: out.remove(pred); out.insert(0,pred)
        return out

    def run(self,ctx):
        hybrid=self.hybrid(ctx)
        baseline=self.gate(ctx,hybrid)
        sig=self.pair(self.sig,p1.SIG_PAIR,ctx,baseline)
        out=self.pair(self.gb,p1.GB_PAIR,ctx,sig)
        if out and sig and out[0]!=sig[0] and sig[0]=="gb18030" and out[0]=="utf-8":
            rate=canon.replacement_rate(ctx.data) if self.lazy_rate else canon.replacement_rate(ctx.data)
            if rate>p1.RATE_THRESHOLD: out=sig
        return out

def p2_reference(model_tuple,s,fused_model,fast_ds):
    ctx=p2.build_ctx(model_tuple,s,"combined","fused",fused_model,True)
    return fast_ds.run(ctx)

CANDIDATES=("p2_baseline","lazy_rate","direct_scalar","direct_scalar_lazy")

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
        direct=DirectDownstream(cal,triad_cal,sig_cal,gb_cal,classes,families,True)

        refs=[p2_reference(models[s["route"]],s,fused[s["route"]],p2ds) for s in rows]
        stats_fold={n:{"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}

        for s,ref in zip(rows,refs):
            mt=models[s["route"]]
            # P2 baseline
            ctx2=p2.build_ctx(mt,s,"combined","fused",fused[s["route"]],True)
            outs={
                "p2_baseline":p2ds.run(ctx2),
                "lazy_rate":p2ds.run(ctx2),
            }
            ctxd=build_ctx(mt,s,fused[s["route"]])
            outs["direct_scalar"]=direct.run(ctxd)
            outs["direct_scalar_lazy"]=direct.run(ctxd)
            for name,out in outs.items():
                st=stats_fold[name]; st["n"]+=1; st["hits"]+=int(out and out[0]==s["label"]); st["mismatch"]+=int(list(out)!=list(ref))

        for name in CANDIDATES:
            if stats_fold[name]["mismatch"]: continue
            t0=time.perf_counter_ns(); sink=0
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    mt=models[s["route"]]
                    if name in ("p2_baseline","lazy_rate"):
                        ctx=p2.build_ctx(mt,s,"combined","fused",fused[s["route"]],True)
                        out=p2ds.run(ctx)
                    else:
                        ctx=build_ctx(mt,s,fused[s["route"]])
                        out=direct.run(ctx)
                    sink+=len(out)
            dt=time.perf_counter_ns()-t0
            stats_fold[name]["elapsed"]=dt; stats_fold[name]["calls"]=len(rows)*TIMING_REPEATS; stats_fold[name]["sink"]=sink

        fr={"fold":fold,"n":len(rows),"candidates":{}}
        for name,st in stats_fold.items():
            fr["candidates"][name]={
                "top1":st["hits"]/max(1,st["n"]),
                "mismatch_n":st["mismatch"],
                "ns_per_call":st["elapsed"]/st["calls"] if st["calls"] else None
            }
            for k in ("n","hits","mismatch","elapsed","calls"): pooled[name][k]+=st[k]
        folds.append(fr)

    out={}
    base_ns=None
    for name,st in pooled.items():
        ns=st["elapsed"]/st["calls"] if st["calls"] else None
        out[name]={"n":st["n"],"hits":st["hits"],"top1":st["hits"]/max(1,st["n"]),"mismatch_n":st["mismatch"],"ns_per_call":ns}
        if name=="p2_baseline": base_ns=ns
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p2"]=base_ns/row["ns_per_call"]
            row["runtime_reduction_vs_p2"]=1-row["ns_per_call"]/base_ns
    eligible=[(r["ns_per_call"],n) for n,r in out.items() if r["mismatch_n"]==0 and r["hits"]==365 and r["ns_per_call"] is not None]
    winner=min(eligible)[1] if eligible else None
    print(json.dumps({
        "phase":"p3_allocation_and_lazy_tournament",
        "canonical_target":{"n":418,"hits":365,"top1":365/418},
        "pooled":out,"winner":winner,"folds":folds,
        "timing_repeats":TIMING_REPEATS,
        "collection_stats":dict(stats),
        "decision_rule":"Preserve 365/418 and exact final ranking; choose fastest pooled candidate."
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
