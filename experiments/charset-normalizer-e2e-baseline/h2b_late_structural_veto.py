from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from codeccat import Detector, load_bundle

import charset_external_scorer_domain_shift_ab as base
from h2_fused_candidate_mask_ab import structural_mask, normalize, collect, dedupe


def late_veto(rank, data):
    if not rank:
        return rank
    mask = structural_mask(data)
    top = rank[0]
    if top not in mask or mask[top]:
        return rank
    for cand in rank[1:]:
        if cand not in mask or mask[cand]:
            return (cand,) + tuple(x for x in rank if x != cand)
    return rank


def evaluate(rows, detector):
    base_hits = veto_hits = 0
    changed = beneficial = harmful = neutral = 0
    impossible_top1 = 0
    truth_masked = 0
    by_route = {r: Counter() for r in ("U","N","RL","RH")}

    for s in rows:
        truth = normalize(s["label"])
        rank = detector.rank(s["data"])
        mask = structural_mask(s["data"])
        if truth in mask and not mask[truth]:
            truth_masked += 1

        if rank and rank[0] in mask and not mask[rank[0]]:
            impossible_top1 += 1

        vrank = late_veto(rank, s["data"])
        bh = int(bool(rank) and rank[0] == truth)
        vh = int(bool(vrank) and vrank[0] == truth)
        base_hits += bh
        veto_hits += vh

        r = s["route"]
        by_route[r]["n"] += 1
        by_route[r]["base_hits"] += bh
        by_route[r]["veto_hits"] += vh

        if (rank[0] if rank else None) != (vrank[0] if vrank else None):
            changed += 1
            if vh and not bh:
                beneficial += 1
            elif bh and not vh:
                harmful += 1
            else:
                neutral += 1

    n = len(rows)
    return {
        "n": n,
        "baseline_hits": base_hits,
        "veto_hits": veto_hits,
        "baseline_top1": base_hits / max(1,n),
        "veto_top1": veto_hits / max(1,n),
        "delta_pp": 100.0*(veto_hits-base_hits)/max(1,n),
        "changed_top1": changed,
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral": neutral,
        "impossible_top1_n": impossible_top1,
        "truth_masked_n": truth_masked,
        "routes": {
            r: {
                "n": c["n"],
                "baseline_top1": c["base_hits"]/max(1,c["n"]),
                "veto_top1": c["veto_hits"]/max(1,c["n"]),
                "delta_hits": c["veto_hits"]-c["base_hits"],
            }
            for r,c in by_route.items() if c["n"]
        }
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--expected-model-sha256", required=True)
    args=ap.parse_args()

    model=Path(args.model)
    actual=hashlib.sha256(model.read_bytes()).hexdigest()
    if actual != args.expected_model_sha256:
        raise RuntimeError(f"model SHA mismatch: {actual}")

    detector=Detector(load_bundle(model))
    crawls=("CC-MAIN-2026-39","CC-MAIN-2026-34")
    reports={}
    all_rows=[]
    collection={}
    for crawl in crawls:
        rows,paths,stats=collect(crawl)
        rows=dedupe(rows)
        all_rows.extend(rows)
        collection[crawl]={"n":len(rows),"warc_paths_considered":len(paths),"stats":stats}
        reports[crawl]=evaluate(rows,detector)

    pooled=dedupe(all_rows)
    pooled_eval=evaluate(pooled,detector)

    print(json.dumps({
        "phase":"h2b_late_structural_veto",
        "policy":{
            "r22_used":False,
            "parameters_tuned":False,
            "release_candidate_modified":False,
            "intervention":"only if final top1 is structurally impossible",
            "score_geometry_modified":False,
        },
        "collection":collection,
        "per_crawl":reports,
        "pooled":pooled_eval,
        "gate":{
            "truth_masked_zero":pooled_eval["truth_masked_n"]==0,
            "harmful_zero":pooled_eval["harmful"]==0,
            "cross_crawl_nonnegative":all(v["veto_hits"]>=v["baseline_hits"] for v in reports.values()),
        }
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
