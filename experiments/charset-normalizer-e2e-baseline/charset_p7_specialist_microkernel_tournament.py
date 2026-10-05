from __future__ import annotations

import argparse
import json
import math
import time

import joblib

import charset_triad_specialist_ab as tri
import charset_p1_cached_inference_context_ab as p1
import charset_p2_multihop_kernel_tournament as p2
import charset_p3_allocation_lazy_tournament as p3
import charset_p6_cached_state_tournament as p6

TIMING_REPEATS=19

class P7Downstream(p6.P6Downstream):
    def __init__(self,*args,scalar_triad=False,inline_pairs=False,**kwargs):
        super().__init__(*args,no_rank_copy=True,direct_membership=True,**kwargs)
        self.scalar_triad=scalar_triad
        self.inline_pairs=inline_pairs

    def triad_logits(self,ctx):
        if not self.scalar_triad:
            return super().triad_logits(ctx)
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
        z0=float(b[0]); z1=float(b[1]); z2=float(b[2])
        for j,x in enumerate(xs):
            z0+=float(w[0,j])*x
            z1+=float(w[1,j])*x
            z2+=float(w[2,j])*x
        ridx=self.route_weight_idx[ctx.route][2]
        z0+=float(w[0,ridx]); z1+=float(w[1,ridx]); z2+=float(w[2,ridx])
        return classes,(z0,z1,z2)

    @staticmethod
    def _in3(rank,label):
        return rank[0]==label or rank[1]==label or rank[2]==label

    def _pair_unchecked(self,fused,idxs,ctx,baseline):
        classes,z=self.pair_logit_indexed(fused,idxs,ctx)
        pred=classes[1] if z>0.0 else classes[0]
        if not self._in3(ctx.rank,pred):
            return baseline
        out=list(baseline)
        if pred in out:
            out.remove(pred); out.insert(0,pred)
        return out

    def run(self,ctx):
        if not self.inline_pairs:
            return super().run(ctx)

        hybrid=self.hybrid(ctx)
        baseline=self.gate(ctx,hybrid)
        r=ctx.rank

        sig=baseline
        if (
            self.sig is not None
            and baseline
            and baseline[0] in p1.SIG_PAIR
            and self._in3(r,p1.SIG_PAIR[0])
            and self._in3(r,p1.SIG_PAIR[1])
        ):
            sig=self._pair_unchecked(self.sig,self.sig_idx[ctx.route],ctx,baseline)

        out=sig
        if (
            self.gb is not None
            and sig
            and sig[0] in p1.GB_PAIR
            and self._in3(r,p1.GB_PAIR[0])
            and self._in3(r,p1.GB_PAIR[1])
        ):
            out=self._pair_unchecked(self.gb,self.gb_idx[ctx.route],ctx,sig)

        if out and sig and out[0]!=sig[0] and sig[0]=="gb18030" and out[0]=="utf-8":
            rate=ctx.replacement_rate
            if rate is None:
                import charset_canonical_guarded_final_validation as canon
                rate=canon.replacement_rate(ctx.data)
            if rate>p1.RATE_THRESHOLD:
                out=sig
        return out

CANDIDATES={
    "p6_baseline":dict(scalar=False,inline=False),
    "scalar_triad":dict(scalar=True,inline=False),
    "inline_pairs":dict(scalar=False,inline=True),
    "combined":dict(scalar=True,inline=True),
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
            name:P7Downstream(
                cal,triad_cal,sig_cal,gb_cal,classes,families,models,True,
                scalar_triad=cfg["scalar"],inline_pairs=cfg["inline"],
            )
            for name,cfg in CANDIDATES.items()
        }

        refs=[]
        for s in rows:
            ctx=p3.build_ctx(models[s["route"]],s,fused[s["route"]],True)
            refs.append(dsmap["p6_baseline"].run(ctx))

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
    base_ns=out["p6_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p6"]=base_ns/row["ns_per_call"]
            row["runtime_reduction_vs_p6"]=1-row["ns_per_call"]/base_ns

    eligible=[(r["ns_per_call"],n) for n,r in out.items()
              if r["mismatch_n"]==0 and r["hits"]==365 and r["ns_per_call"] is not None]
    winner=min(eligible)[1] if eligible else None

    print(json.dumps({
        "phase":"p7_specialist_microkernel_tournament",
        "canonical_target":{"n":418,"hits":365,"top1":365/418},
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "timing_repeats":TIMING_REPEATS,
        "pooled":out,"winner":winner,"folds":folds,
        "decision_rule":"Preserve exact P6 ranking and 365/418; choose fastest same-run candidate."
    },indent=2,sort_keys=True))

if __name__=="__main__":
    main()
