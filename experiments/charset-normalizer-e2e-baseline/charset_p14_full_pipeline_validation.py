from __future__ import annotations

import argparse
import json
import time

import joblib

import charset_p2_multihop_kernel_tournament as p2
import charset_p8_context_metadata_tournament as p8
import charset_p12_normalized_count_tournament as p12
import charset_p14_calibrator_hotpath_tournament as p14

TIMING_REPEATS = 31


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()
    state = joblib.load(args.state)
    assert state["schema_version"] == 1
    assert sum(len(f["rows"]) for f in state["folds"]) == 418

    classes, families = state["classes"], state["families"]
    pooled = {
        "p12_baseline": {"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0},
        "p14_combined": {"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0},
    }
    folds = []

    for fd in state["folds"]:
        models, rows = fd["models"], fd["rows"]
        fused = {r: p2.fuse_scaler_linear(mt) for r, mt in models.items()}
        meta = {r: p8.RouteMeta(mt) for r, mt in models.items()}
        base = p14.CalDownstream(
            fd["cal"], fd["triad_cal"], fd["sig_cal"], fd["gb_cal"],
            classes, families, models, True,
            scalar_triad=True, inline_pairs=False, mode="baseline",
        )
        opt = p14.CalDownstream(
            fd["cal"], fd["triad_cal"], fd["sig_cal"], fd["gb_cal"],
            classes, families, models, True,
            scalar_triad=True, inline_pairs=False, mode="combined",
        )

        def make_ctx(s):
            return p12.build_ctx(
                s, fused[s["route"]], meta[s["route"]],
                {"uni_mode":"assign_multiply","bi_mode":"assign_multiply"},
            )

        refs = []
        for s in rows:
            refs.append(base.run(make_ctx(s)))

        fs = {
            "p12_baseline": {"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0},
            "p14_combined": {"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0},
        }
        for s, ref in zip(rows, refs):
            for name, ds in (("p12_baseline", base), ("p14_combined", opt)):
                out = ds.run(make_ctx(s))
                st = fs[name]
                st["n"] += 1
                st["hits"] += int(out and out[0] == s["label"])
                st["mismatch"] += int(list(out) != list(ref))

        for name, ds in (("p12_baseline", base), ("p14_combined", opt)):
            if fs[name]["mismatch"]:
                continue
            sink = 0
            t0 = time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    out = ds.run(make_ctx(s))
                    sink += len(out)
            fs[name]["elapsed"] = time.perf_counter_ns() - t0
            fs[name]["calls"] = len(rows) * TIMING_REPEATS
            fs[name]["sink"] = sink

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
            "n":st["n"],"hits":st["hits"],"top1":st["hits"]/max(1,st["n"]),
            "mismatch_n":st["mismatch"],"ns_per_call":ns,
        }
    base_ns=out["p12_baseline"]["ns_per_call"]
    opt_ns=out["p14_combined"]["ns_per_call"]
    out["p14_combined"]["speedup_vs_p12"]=base_ns/opt_ns
    out["p14_combined"]["runtime_reduction_vs_p12"]=1-opt_ns/base_ns

    print(json.dumps({
        "phase":"p14_full_pipeline_validation",
        "canonical_target":{"n":418,"hits":365,"top1":365/418},
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "timing_repeats":TIMING_REPEATS,
        "pooled":out,"folds":folds,
        "accepted": (
            out["p14_combined"]["hits"]==365
            and out["p14_combined"]["mismatch_n"]==0
            and opt_ns < base_ns
        ),
        "decision_rule":"Lock P14 only if full pipeline preserves 365/418, exact final ranking, and beats locked P12 in the same run."
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
