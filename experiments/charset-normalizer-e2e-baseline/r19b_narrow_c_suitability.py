from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from contextlib import contextmanager

import joblib
import numpy as np

import charset_external_scorer_domain_shift_ab as base
import charset_external_scorer_learning_curve as lc
import r13_representation_sufficiency as r13
import r16a_fused_full_b as r16
import r17b_ambiguity_gated_full_b as r17

OUTER_FOLDS=4
TRAIN_N=300
R12_THRESHOLD=0.65
C_VALUES=(0.40,0.50,0.60)
CENTER=0.50

ORIG_LC_FIT=lc.fit_route_models


def fit_route_models_c(c):
    def _fit(ext_rows, legacy_X, legacy_y, legacy_b):
        models={}
        for route in ("U","N","RL","RH"):
            li=np.where(legacy_b==route)[0]
            Xtr=legacy_X[li]
            ytr=legacy_y[li]

            er=[s for s in ext_rows if s["route"]==route]
            if er:
                Xext=np.stack([base.scorer_features(s["data"]) for s in er])
                yext=np.array([s["label"] for s in er],dtype=object)
                known=np.isin(yext,np.unique(legacy_y))
                if known.any():
                    Xtr=np.concatenate([Xtr,Xext[known]],axis=0)
                    ytr=np.concatenate([ytr,yext[known]],axis=0)

            if len(Xtr)>=20 and len(np.unique(ytr))>=2:
                models[route]=base.fit_linear(Xtr,ytr,C=c)
        return models
    return _fit


@contextmanager
def route_model_c(c):
    lc.fit_route_models=fit_route_models_c(c)
    try:
        yield
    finally:
        lc.fit_route_models=ORIG_LC_FIT


def eval_variant(c,external,fold_assign,base_X,full_X,legacy_y,legacy_b,full_y,full_b):
    total_b=Counter(); total_f=Counter(); folds=[]
    for fold in range(OUTER_FOLDS):
        train_pool=[s for s in external if fold_assign[s["warc_path"]]!=fold]
        test_rows=[s for s in external if fold_assign[s["warc_path"]]==fold]
        with route_model_c(c):
            bstack=r17.fit_stack(r13.baseline_features,train_pool,base_X,legacy_y,legacy_b,True)
            fstack=r17.fit_stack(r16.fused_full_b,train_pool,full_X,full_y,full_b,False)

        fb=Counter(); ff=Counter()
        for s in test_rows:
            bout,_=r17.predict(bstack,s,True)
            if bout is None:
                continue
            truth=s["label"]
            bhit=int(bout and bout[0]==truth)
            total_b["n"]+=1; total_b["hits"]+=bhit
            total_b[f"{s['route']}_n"]+=1; total_b[f"{s['route']}_hits"]+=bhit
            fb["n"]+=1; fb["hits"]+=bhit

            fout=None
            if s["route"] in ("U","RH"):
                fout,_=r17.predict(fstack,s,False)
            chosen=fout if fout is not None else bout
            fhit=int(chosen and chosen[0]==truth)
            total_f["n"]+=1; total_f["hits"]+=fhit
            total_f[f"{s['route']}_n"]+=1; total_f[f"{s['route']}_hits"]+=fhit
            ff["n"]+=1; ff["hits"]+=fhit
        folds.append({"fold":fold,"baseline_hits":fb["hits"],"full_hits":ff["hits"],"n":fb["n"]})

    def pack(x):
        return {
            "n":x["n"],"hits":x["hits"],"top1":x["hits"]/max(1,x["n"]),
            "routes":{
                r:{
                    "n":x[f"{r}_n"],
                    "hits":x[f"{r}_hits"],
                    "top1":x[f"{r}_hits"]/max(1,x[f"{r}_n"])
                }
                for r in ("U","N","RL","RH") if x[f"{r}_n"]
            }
        }
    return {"baseline":pack(total_b),"full":pack(total_f),"folds":folds}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True); args=ap.parse_args()
    state=joblib.load(args.state)
    external=r17.reconstruct_external(state); legacy_rows=r17.raw_legacy_rows()
    base_X,legacy_y,legacy_b=r17.legacy_arrays(r13.baseline_features,legacy_rows)
    full_X,full_y,full_b=r17.legacy_arrays(r16.fused_full_b,legacy_rows)

    paths=sorted(set(s["warc_path"] for s in external))
    fold_assign={p:int.from_bytes(hashlib.sha256(("learning-curve:"+p).encode()).digest()[:8],"big")%OUTER_FOLDS for p in paths}

    # Exact reference using the untouched production-research fit function.
    reference=eval_variant(CENTER,external,fold_assign,base_X,full_X,legacy_y,legacy_b,full_y,full_b)

    results={}
    for c in C_VALUES:
        results[str(c)]=eval_variant(c,external,fold_assign,base_X,full_X,legacy_y,legacy_b,full_y,full_b)

    center=results[str(CENTER)]
    reproduction={
        "reference_baseline_hits":reference["baseline"]["hits"],
        "center_baseline_hits":center["baseline"]["hits"],
        "reference_full_hits":reference["full"]["hits"],
        "center_full_hits":center["full"]["hits"],
        "exact_match":(
            reference["baseline"]["hits"]==center["baseline"]["hits"]
            and reference["full"]["hits"]==center["full"]["hits"]
            and reference["folds"]==center["folds"]
        )
    }

    def plateau(key):
        hits=[results[str(c)][key]["hits"] for c in C_VALUES]
        ch=center[key]["hits"]
        return {
            "center_hits":ch,
            "min_hits":min(hits),
            "max_hits":max(hits),
            "range_hits":max(hits)-min(hits),
            "center_within_one_hit_of_best":max(hits)-ch<=1,
            "plateau_within_two_hits_total":max(hits)-min(hits)<=2,
        }

    print(json.dumps({
        "phase":"r19b_narrow_c_suitability",
        "purpose":"Validate local robustness of the existing C=0.5 choice; do not select a new C from development accuracy.",
        "values":C_VALUES,
        "center":CENTER,
        "reference_reproduction":reproduction,
        "results":results,
        "baseline_plateau":plateau("baseline"),
        "full_plateau":plateau("full"),
        "acceptance_rule":"Interpret C=0.5 as locally suitable only if exact central reproduction passes, center is within one hit of local best, three-point range is <=2 hits, and no route/fold shows a sharp collapse.",
        "methodology":{
            "only_route_scorer_C_changes":True,
            "all_other_downstream_hyperparameters_locked":True,
            "same_outer_folds":True,
            "same_training_corpus":True,
            "no_new_features":True,
            "no_new_model_family":True,
            "cc_main_2026_30_used":False
        }
    },indent=2,sort_keys=True))


if __name__=="__main__": main()
