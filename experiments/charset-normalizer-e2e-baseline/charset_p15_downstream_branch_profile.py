from __future__ import annotations

import argparse
import json

import joblib

import charset_p1_cached_inference_context_ab as p1
import charset_p2_multihop_kernel_tournament as p2
import charset_p7_specialist_microkernel_tournament as p7
import charset_p8_context_metadata_tournament as p8
import charset_p14_feature_linear_boundary_tournament as p14
import charset_triad_specialist_ab as tri

LOCKED_CFG = {"f64": False, "gemv": True}


def pct(n, d):
    return n / d if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()

    state = joblib.load(args.state)
    assert state["schema_version"] == 1
    assert sum(len(f["rows"]) for f in state["folds"]) == 418

    classes, families = state["classes"], state["families"]
    total = {
        "n": 0,
        "rule_changes_top": 0,
        "cal_calls": 0,
        "cal_changes_top": 0,
        "triad_guard_top": 0,
        "triad_guard_top3_count": 0,
        "triad_changes_top": 0,
        "sig_guard_top": 0,
        "sig_guard_pair_top3": 0,
        "sig_changes_top": 0,
        "gb_guard_top": 0,
        "gb_guard_pair_top3": 0,
        "gb_changes_top": 0,
        "replacement_guard": 0,
        "replacement_rate_computed": 0,
        "instrumented_mismatch": 0,
    }
    folds = []

    for fd in state["folds"]:
        models, rows = fd["models"], fd["rows"]
        fused = {r: p2.fuse_scaler_linear(mt) for r, mt in models.items()}
        meta = {r: p8.RouteMeta(mt) for r, mt in models.items()}
        ds = p7.P7Downstream(
            fd["cal"], fd["triad_cal"], fd["sig_cal"], fd["gb_cal"],
            classes, families, models, True,
            scalar_triad=True, inline_pairs=False,
        )

        fs = {k: 0 for k in total}
        fs["n"] = len(rows)

        for s in rows:
            ctx = p14.build_ctx(
                s, fused[s["route"]], meta[s["route"]], LOCKED_CFG
            )
            canonical = ds.run(ctx)

            # Hybrid stage: measure the rule fast-path and whether calibration
            # is still needed.
            rule = p1.cached_rule_rerank(ctx)
            rule_changed = bool(rule and ctx.rank and rule[0] != ctx.rank[0])
            if rule_changed:
                fs["rule_changes_top"] += 1
                hybrid = rule
            else:
                fs["cal_calls"] += 1
                cal = ds.choose_cal(ctx)
                if cal and ctx.rank and cal[0] != ctx.rank[0]:
                    fs["cal_changes_top"] += 1
                hybrid = cal

            # Triad gate.
            if ds.triad is not None and hybrid and hybrid[0] in tri.TRIAD:
                fs["triad_guard_top"] += 1
                r = ctx.rank
                count = (
                    int(r[0] in tri.TRIAD)
                    + int(r[1] in tri.TRIAD)
                    + int(r[2] in tri.TRIAD)
                )
                if count >= 2:
                    fs["triad_guard_top3_count"] += 1
            baseline = ds.gate(ctx, hybrid)
            if baseline and hybrid and baseline[0] != hybrid[0]:
                fs["triad_changes_top"] += 1

            # SIG pair.
            sig_guard = bool(
                ds.sig is not None and baseline and baseline[0] in p1.SIG_PAIR
            )
            if sig_guard:
                fs["sig_guard_top"] += 1
                r = ctx.rank
                if ds._in3(r, p1.SIG_PAIR[0]) and ds._in3(r, p1.SIG_PAIR[1]):
                    fs["sig_guard_pair_top3"] += 1
            sig = ds.pair(ds.sig, p1.SIG_PAIR, ctx, baseline)
            if sig and baseline and sig[0] != baseline[0]:
                fs["sig_changes_top"] += 1

            # GB pair.
            gb_guard = bool(
                ds.gb is not None and sig and sig[0] in p1.GB_PAIR
            )
            if gb_guard:
                fs["gb_guard_top"] += 1
                r = ctx.rank
                if ds._in3(r, p1.GB_PAIR[0]) and ds._in3(r, p1.GB_PAIR[1]):
                    fs["gb_guard_pair_top3"] += 1
            out = ds.pair(ds.gb, p1.GB_PAIR, ctx, sig)
            if out and sig and out[0] != sig[0]:
                fs["gb_changes_top"] += 1

            guard = bool(
                out
                and sig
                and out[0] != sig[0]
                and sig[0] == "gb18030"
                and out[0] == "utf-8"
            )
            if guard:
                fs["replacement_guard"] += 1
                if ctx.replacement_rate is None:
                    fs["replacement_rate_computed"] += 1

            if list(out) != list(canonical):
                fs["instrumented_mismatch"] += 1

        for k, v in fs.items():
            total[k] += v if k != "n" else 0
        total["n"] += len(rows)

        folds.append({"fold": fd["fold"], "n": len(rows), "counts": fs})

    n = total["n"]
    rates = {k: pct(v, n) for k, v in total.items() if k not in ("n", "instrumented_mismatch")}

    print(json.dumps({
        "phase": "p15_downstream_branch_frequency_profile",
        "cache_schema": state["schema_version"],
        "corpus_fingerprint": state["corpus_fingerprint"],
        "counts": total,
        "rates_per_sample": rates,
        "folds": folds,
        "interpretation": (
            "Use branch incidence to choose P15/P16 no-op fast paths; "
            "instrumented stage composition must exactly match canonical ds.run."
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
