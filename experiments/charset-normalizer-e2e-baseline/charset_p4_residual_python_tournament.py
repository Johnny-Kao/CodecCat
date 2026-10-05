from __future__ import annotations

import hashlib, json, math, time
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

OUTER_FOLDS=4
TRAIN_N=300
TIMING_REPEATS=13

class LiteCtx:
    __slots__=("data","route","rank","top_scores","raw_scores","classes",
               "has_utf8_bom","strict_utf8","ascii_only",
               "high_byte_ratio","nul_ratio","sample_len_4k","replacement_rate")
    def __init__(self,data,route,rank,top_scores,raw_scores,classes,bom,strict,ascii_only,high,nul,n4k):
        self.data=data; self.route=route; self.rank=rank; self.top_scores=top_scores
        self.raw_scores=raw_scores; self.classes=classes
        self.has_utf8_bom=bom; self.strict_utf8=strict; self.ascii_only=ascii_only
        self.high_byte_ratio=high; self.nul_ratio=nul; self.sample_len_4k=n4k
        self.replacement_rate=None

def build_lite(model_tuple,s,fused_model):
    x=p2.features_combined(s["data"])
    raw=p2.fused_raw(fused_model,x)
    _,model=model_tuple
    order=np.argsort(raw)[::-1]
    rank=tuple(model.classes_[order])
    sample=s["data"][:4096]
    arr=np.frombuffer(sample,dtype=np.uint8)
    strict=False
    try:
        s["data"].decode("utf-8",errors="strict"); strict=True
    except UnicodeDecodeError:
        pass
    bom=s["data"].startswith(b"\xef\xbb\xbf")
    if len(arr):
        high_count=int(np.count_nonzero(arr>=128)); nul_count=int(np.count_nonzero(arr==0))
        high=high_count/len(arr); nul=nul_count/len(arr); ascii_flag=(high_count==0)
    else:
        high=nul=0.0; ascii_flag=True
    return LiteCtx(
        s["data"],s["route"],rank,np.asarray(raw)[order[:3]],np.asarray(raw),
        tuple(model.classes_),bom,strict,ascii_flag,high,nul,len(arr)
    )

def move_front(rank,pred):
    if not rank or rank[0]==pred: return list(rank)
    try: i=rank.index(pred)
    except ValueError: return list(rank)
    return [rank[i],*rank[:i],*rank[i+1:]]

class P4Downstream(p3c.IndexedDirectDownstream):
    def __init__(self,*args,manual_smallops=False,no_list_remove=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.manual_smallops=manual_smallops
        self.no_list_remove=no_list_remove

    def choose_cal(self,ctx):
        if self.cal is None or len(ctx.rank)<2: return list(ctx.rank)
        top=float(ctx.top_scores[0] if hasattr(ctx,"top_scores") else ctx.sorted_scores[0])
        second=float(ctx.top_scores[1] if hasattr(ctx,"top_scores") else ctx.sorted_scores[1])
        scores=ctx.top_scores if hasattr(ctx,"top_scores") else ctx.sorted_scores
        cand=ctx.rank[:calmod.TOP_CANDIDATES]
        z0=self.cal_logit(ctx,cand[0],0,float(scores[0]),top,second)
        z1=self.cal_logit(ctx,cand[1],1,float(scores[1]),top,second)
        z2=self.cal_logit(ctx,cand[2],2,float(scores[2]),top,second)
        if self.manual_smallops:
            winner=0
            best=z0
            if z1>best: winner=1; best=z1
            if z2>best: winner=2
        else:
            winner=int(np.argmax((z0,z1,z2)))
        if winner==0: return list(ctx.rank)
        chosen=cand[winner]
        if self.no_list_remove:
            return move_front(ctx.rank,chosen)
        out=list(ctx.rank)
        if chosen in out:
            out.remove(chosen)
            out.insert(0,chosen)
        return out

    def gate(self,ctx,hybrid):
        if self.triad is None or not hybrid or hybrid[0] not in tri.TRIAD: return hybrid
        top3=ctx.rank[:3]
        if sum(c in tri.TRIAD for c in top3)<2: return hybrid
        classes,logits=self.triad_logits(ctx)
        if self.manual_smallops:
            a=float(logits[0]); b=float(logits[1]); c=float(logits[2])
            if a>=b and a>=c: idx=0; m=a; o1=b; o2=c
            elif b>=c: idx=1; m=b; o1=a; o2=c
            else: idx=2; m=c; o1=a; o2=b
            # pmax >= .5 iff exp(o1-m)+exp(o2-m) <= 1
            p_ok=(math.exp(o1-m)+math.exp(o2-m))<=1.0
        else:
            idx=int(np.argmax(logits))
            shifted=logits-np.max(logits); ex=np.exp(shifted)
            p_ok=float(ex[idx]/ex.sum())>=0.50
        pred=classes[idx]
        if not p_ok or pred not in top3: return hybrid
        if self.no_list_remove: return move_front(hybrid,pred)
        out=list(hybrid)
        if pred in out: out.remove(pred); out.insert(0,pred)
        return out

    def pair(self,fused,pair,ctx,baseline):
        if fused is None or not baseline or baseline[0] not in pair: return baseline
        top3=ctx.rank[:3]
        if pair[0] not in top3 or pair[1] not in top3: return baseline
        idxs=self.sig_idx[ctx.route] if pair==p1.SIG_PAIR else self.gb_idx[ctx.route]
        classes,z=self.pair_logit_indexed(fused,idxs,ctx)
        pred=classes[1] if z>0.0 else classes[0]
        if pred not in top3: return baseline
        if self.no_list_remove: return move_front(baseline,pred)
        out=list(baseline)
        if pred in out: out.remove(pred); out.insert(0,pred)
        return out

def p3_ref(model_tuple,s,fused_model,ds):
    ctx=p3.build_ctx(model_tuple,s,fused_model,True)
    return ds.run(ctx)

CANDIDATES={
    "p3_baseline":dict(lite=False,manual=False,nolist=False),
    "lite_ctx":dict(lite=True,manual=False,nolist=False),
    "manual_smallops":dict(lite=False,manual=True,nolist=False),
    "no_list_remove":dict(lite=False,manual=False,nolist=True),
    "lite_manual":dict(lite=True,manual=True,nolist=False),
    "all_p4":dict(lite=True,manual=True,nolist=True),
}

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
        refds=p3c.IndexedDirectDownstream(cal,triad_cal,sig_cal,gb_cal,classes,families,models,True)
        refs=[p3_ref(models[s["route"]],s,fused[s["route"]],refds) for s in rows]

        dsmap={
          n:P4Downstream(cal,triad_cal,sig_cal,gb_cal,classes,families,models,True,
                         manual_smallops=cfg["manual"],no_list_remove=cfg["nolist"])
          for n,cfg in CANDIDATES.items()
        }
        fs={n:{"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}

        for s,ref in zip(rows,refs):
            mt=models[s["route"]]
            for name,cfg in CANDIDATES.items():
                ctx=build_lite(mt,s,fused[s["route"]]) if cfg["lite"] else p3.build_ctx(mt,s,fused[s["route"]],True)
                out=dsmap[name].run(ctx)
                st=fs[name]; st["n"]+=1; st["hits"]+=int(out and out[0]==s["label"]); st["mismatch"]+=int(list(out)!=list(ref))

        for name,cfg in CANDIDATES.items():
            if fs[name]["mismatch"]: continue
            t0=time.perf_counter_ns(); sink=0
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    mt=models[s["route"]]
                    ctx=build_lite(mt,s,fused[s["route"]]) if cfg["lite"] else p3.build_ctx(mt,s,fused[s["route"]],True)
                    out=dsmap[name].run(ctx); sink+=len(out)
            dt=time.perf_counter_ns()-t0
            fs[name]["elapsed"]=dt; fs[name]["calls"]=len(rows)*TIMING_REPEATS; fs[name]["sink"]=sink

        fr={"fold":fold,"n":len(rows),"candidates":{}}
        for name,st in fs.items():
            fr["candidates"][name]={"top1":st["hits"]/max(1,st["n"]),"mismatch_n":st["mismatch"],
                "ns_per_call":st["elapsed"]/st["calls"] if st["calls"] else None}
            for k in ("n","hits","mismatch","elapsed","calls"): pooled[name][k]+=st[k]
        folds.append(fr)

    out={}
    for name,st in pooled.items():
        ns=st["elapsed"]/st["calls"] if st["calls"] else None
        out[name]={"n":st["n"],"hits":st["hits"],"top1":st["hits"]/max(1,st["n"]),"mismatch_n":st["mismatch"],"ns_per_call":ns}
    base_ns=out["p3_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p3"]=base_ns/row["ns_per_call"]
            row["runtime_reduction_vs_p3"]=1-row["ns_per_call"]/base_ns
    eligible=[(r["ns_per_call"],n) for n,r in out.items() if r["mismatch_n"]==0 and r["hits"]==365 and r["ns_per_call"] is not None]
    winner=min(eligible)[1] if eligible else None
    print(json.dumps({
      "phase":"p4_residual_python_multihop_tournament",
      "canonical_target":{"n":418,"hits":365,"top1":365/418},
      "timing_repeats":TIMING_REPEATS,"pooled":out,"winner":winner,"folds":folds,
      "collection_stats":dict(stats),
      "decision_rule":"Preserve 365/418 and exact P3 final ranking; choose fastest pooled candidate."
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
