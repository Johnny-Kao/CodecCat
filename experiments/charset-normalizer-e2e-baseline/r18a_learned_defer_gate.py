from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import charset_candidate_calibration_ab as calmod
import r13_representation_sufficiency as r13
import r16a_fused_full_b as r16
import r17b_ambiguity_gated_full_b as r17

OUTER_FOLDS=4
INNER_FOLDS=2
BUDGETS=(0.25,0.40,0.50,0.60)
FAMILIES=("ascii","cjk","koi8","other","singlebyte-iso","singlebyte-win","utf")


def inner_assign(path):
    return int.from_bytes(hashlib.sha256(("r18a-defer:"+path).encode()).digest()[:8],"big")%INNER_FOLDS


def gate_feature(stack,s):
    def _calc():
        model=stack["models"].get(s["route"])
        if model is None:
            return None
        rank,scores=calmod.raw_rank_scores(model,s["data"])
        if len(scores)<2:
            return None
        vals=np.asarray(scores[:min(5,len(scores))],dtype=np.float64)
        z=vals-np.max(vals)
        p=np.exp(z); p/=max(1e-30,float(np.sum(p)))
        entropy=float(-np.sum(p*np.log(np.maximum(p,1e-30))))
        top1=float(scores[0]); top2=float(scores[1])
        top3=float(scores[2]) if len(scores)>=3 else top2
        fam=calmod.fam(str(rank[0])) if rank else "other"
        a=np.frombuffer(s["data"][:4096],dtype=np.uint8)
        high=float(np.mean(a>=128)) if len(a) else 0.0
        printable=float(np.mean((a>=32)&(a<=126))) if len(a) else 0.0
        route_oh=[1.0 if s["route"]==x else 0.0 for x in ("U","RH")]
        fam_oh=[1.0 if fam==x else 0.0 for x in FAMILIES]
        return np.asarray([
            top1,top2,top1-top2,top1-top3,entropy,
            high,printable,math.log2(len(a)+1)/16.0,
            *route_oh,*fam_oh
        ],dtype=np.float32)
    return r17.with_scorer(stack["scorer"],_calc)


def build_gate_training(train_pool,base_X,legacy_y,legacy_b,full_X,full_y,full_b):
    X=[]; y=[]; sw=[]; meta=Counter()
    for inner in range(INNER_FOLDS):
        itr=[s for s in train_pool if inner_assign(s["warc_path"])!=inner]
        iva=[s for s in train_pool if inner_assign(s["warc_path"])==inner and s["route"] in ("U","RH")]
        if not iva:
            continue
        bstack=r17.fit_stack(r13.baseline_features,itr,base_X,legacy_y,legacy_b,False)
        fstack=r17.fit_stack(r16.fused_full_b,itr,full_X,full_y,full_b,False)
        for s in iva:
            feat=gate_feature(bstack,s)
            if feat is None:
                continue
            bout,_=r17.predict(bstack,s,False)
            fout,_=r17.predict(fstack,s,False)
            if not bout or not fout:
                continue
            truth=s["label"]
            b=int(bout[0]==truth); f=int(fout[0]==truth)
            benefit=int(f>b)
            harmful=int(f<b)
            X.append(feat); y.append(benefit); sw.append(2.0 if harmful else 1.0)
            meta["n"]+=1; meta["benefit"]+=benefit; meta["harmful"]+=harmful; meta["neutral"]+=int(f==b)
    if len(set(y))<2:
        return None,meta
    X=np.stack(X); y=np.asarray(y,dtype=np.int32); sw=np.asarray(sw,dtype=np.float64)
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    clf=LogisticRegression(max_iter=2500,C=0.5,solver="lbfgs",class_weight="balanced")
    clf.fit(Xs,y,sample_weight=sw)
    probs=clf.predict_proba(Xs)[:,1]
    thresholds={}
    for budget in BUDGETS:
        thresholds[budget]=float(np.quantile(probs,1.0-budget))
    return (sc,clf,thresholds),meta


def pack(c,baseline_hits):
    return {
        "n":c["n"],"hits":c["hits"],"top1":c["hits"]/max(1,c["n"]),
        "delta_hits_vs_baseline":c["hits"]-baseline_hits,
        "eligible":c["eligible"],"escalated":c["escalated"],
        "escalation_rate_all":c["escalated"]/max(1,c["n"]),
        "escalation_rate_eligible":c["escalated"]/max(1,c["eligible"]),
        "beneficial":c["beneficial"],"harmful":c["harmful"],"neutral_change":c["neutral_change"],
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    external=r17.reconstruct_external(state); legacy_rows=r17.raw_legacy_rows()
    base_X,legacy_y,legacy_b=r17.legacy_arrays(r13.baseline_features,legacy_rows)
    full_X,full_y,full_b=r17.legacy_arrays(r16.fused_full_b,legacy_rows)

    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}

    baseline=Counter(); full=Counter(); arms={b:Counter() for b in BUDGETS}; folds=[]

    for fold in range(OUTER_FOLDS):
        train_pool=[s for s in external if fold_assign[s["warc_path"]]!=fold]
        test_rows=[s for s in external if fold_assign[s["warc_path"]]==fold]
        gate,meta=build_gate_training(train_pool,base_X,legacy_y,legacy_b,full_X,full_y,full_b)
        bstack=r17.fit_stack(r13.baseline_features,train_pool,base_X,legacy_y,legacy_b,True)
        fstack=r17.fit_stack(r16.fused_full_b,train_pool,full_X,full_y,full_b,False)

        fc={b:Counter() for b in BUDGETS}; fb=Counter(); ff=Counter()
        for s in test_rows:
            bout,_=r17.predict(bstack,s,True)
            if not bout:
                continue
            truth=s["label"]; bhit=int(bout[0]==truth)
            baseline["n"]+=1; baseline["hits"]+=bhit
            fb["n"]+=1; fb["hits"]+=bhit

            fout=None
            if s["route"] in ("U","RH"):
                fout,_=r17.predict(fstack,s,False)
            fullout=fout if fout is not None else bout
            fhit=int(fullout and fullout[0]==truth)
            full["n"]+=1; full["hits"]+=fhit
            ff["n"]+=1; ff["hits"]+=fhit

            feat=gate_feature(bstack,s) if s["route"] in ("U","RH") else None
            prob=None
            if gate is not None and feat is not None:
                sc,clf,_=gate
                prob=float(clf.predict_proba(sc.transform(feat[None,:]))[0,1])

            for budget in BUDGETS:
                c=arms[budget]; d=fc[budget]
                c["n"]+=1; d["n"]+=1
                if s["route"] in ("U","RH"):
                    c["eligible"]+=1; d["eligible"]+=1
                escalate=False
                if gate is not None and prob is not None and s["route"] in ("U","RH"):
                    escalate=prob>=gate[2][budget]
                out=fullout if escalate else bout
                hit=int(out and out[0]==truth)
                c["hits"]+=hit; d["hits"]+=hit
                if escalate:
                    c["escalated"]+=1; d["escalated"]+=1
                    if hit>bhit: c["beneficial"]+=1; d["beneficial"]+=1
                    elif hit<bhit: c["harmful"]+=1; d["harmful"]+=1
                    elif fullout and bout and fullout[0]!=bout[0]:
                        c["neutral_change"]+=1; d["neutral_change"]+=1

        folds.append({
            "fold":fold,
            "gate_training":dict(meta),
            "gate_thresholds":({str(k):v for k,v in gate[2].items()} if gate else {}),
            "baseline_hits":fb["hits"],"full_hits":ff["hits"],
            "arms":{str(b):pack(fc[b],fb["hits"]) for b in BUDGETS},
        })

    print(json.dumps({
        "phase":"r18a_learned_defer_gate",
        "baseline":{"n":baseline["n"],"hits":baseline["hits"],"top1":baseline["hits"]/max(1,baseline["n"])},
        "full_all":{"n":full["n"],"hits":full["hits"],"top1":full["hits"]/max(1,full["n"]),"delta_hits_vs_baseline":full["hits"]-baseline["hits"]},
        "arms":{str(b):pack(arms[b],baseline["hits"]) for b in BUDGETS},
        "folds":folds,
        "gate_features":["top1","top2","margin12","margin13","top5_entropy","high_byte_ratio","printable_ascii_ratio","log_sample_len","route_U","route_RH","predicted_family_onehot"],
        "training_target":"defer=1 only when full-B fixes a baseline error; harmful full-B changes are weighted 2x as negative examples.",
        "acceptance_rule":"Prefer any arm that preserves full-all hits with fewer escalations than the R17B 250/418 margin-gate point; otherwise compare equal-budget hits.",
        "methodology":{"gate_labels_inner_crossfit":True,"outer_test_not_used_for_gate_fit_or_threshold":True,"cc_main_2026_30_used":False}
    },indent=2,sort_keys=True))


if __name__=="__main__": main()
