from __future__ import annotations

import argparse
import json
import time

import joblib

import charset_p2_multihop_kernel_tournament as p2
import charset_p7_specialist_microkernel_tournament as p7
import charset_p8_context_metadata_tournament as p8
import charset_p12_normalized_count_tournament as p12

REPEATS = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()
    state = joblib.load(args.state)
    assert state["schema_version"] == 1
    assert sum(len(f["rows"]) for f in state["folds"]) == 418

    classes = state["classes"]
    families = state["families"]

    weighted = {k: [0.0, 0] for k in ("hybrid", "gate", "sig", "gb", "run")}
    activations = {
        "rule_override": 0,
        "triad_entry": 0,
        "triad_changed": 0,
        "sig_entry": 0,
        "sig_changed": 0,
        "gb_entry": 0,
        "gb_changed": 0,
        "replacement_guard_shape": 0,
        "n": 0,
    }
    fold_rows = []

    for fd in state["folds"]:
        models = fd["models"]
        rows = fd["rows"]
        fused = {r: p2.fuse_scaler_linear(mt) for r, mt in models.items()}
        meta = {r: p8.RouteMeta(mt) for r, mt in models.items()}
        ds = p7.P7Downstream(
            fd["cal"], fd["triad_cal"], fd["sig_cal"], fd["gb_cal"],
            classes, families, models, True,
            scalar_triad=True, inline_pairs=False,
        )

        ctxs = [
            p12.build_ctx(
                s, fused[s["route"]], meta[s["route"]],
                {"uni_mode": "assign_multiply", "bi_mode": "assign_multiply"},
            )
            for s in rows
        ]

        hybrids = [ds.hybrid(ctx) for ctx in ctxs]
        gates = [ds.gate(ctx, h) for ctx, h in zip(ctxs, hybrids)]
        sigs = [ds.pair(ds.sig, __import__("charset_p1_cached_inference_context_ab").SIG_PAIR, ctx, g)
                for ctx, g in zip(ctxs, gates)]
        gbs = [ds.pair(ds.gb, __import__("charset_p1_cached_inference_context_ab").GB_PAIR, ctx, sig)
               for ctx, sig in zip(ctxs, sigs)]

        import charset_p1_cached_inference_context_ab as p1
        local = {k: 0 for k in activations}
        for ctx, h, g, sig, out in zip(ctxs, hybrids, gates, sigs, gbs):
            local["n"] += 1
            rule = p1.cached_rule_rerank(ctx)
            local["rule_override"] += int(bool(rule and ctx.rank and rule[0] != ctx.rank[0]))
            local["triad_entry"] += int(bool(ds.triad is not None and h and h[0] in __import__("charset_triad_specialist_ab").TRIAD))
            local["triad_changed"] += int(list(g) != list(h))
            local["sig_entry"] += int(bool(ds.sig is not None and g and g[0] in p1.SIG_PAIR))
            local["sig_changed"] += int(list(sig) != list(g))
            local["gb_entry"] += int(bool(ds.gb is not None and sig and sig[0] in p1.GB_PAIR))
            local["gb_changed"] += int(list(out) != list(sig))
            local["replacement_guard_shape"] += int(bool(
                out and sig and out[0] != sig[0]
                and sig[0] == "gb18030" and out[0] == "utf-8"
            ))
        for k, v in local.items():
            activations[k] += v

        n = len(ctxs)

        def bench(fn):
            sink = 0
            t0 = time.perf_counter_ns()
            for _ in range(REPEATS):
                for i in range(n):
                    x = fn(i)
                    sink += len(x) if x is not None else 0
            return (time.perf_counter_ns() - t0) / (REPEATS * n), sink

        stage = {}
        stage["hybrid"], _ = bench(lambda i: ds.hybrid(ctxs[i]))
        stage["gate"], _ = bench(lambda i: ds.gate(ctxs[i], hybrids[i]))
        stage["sig"], _ = bench(lambda i: ds.pair(ds.sig, p1.SIG_PAIR, ctxs[i], gates[i]))
        stage["gb"], _ = bench(lambda i: ds.pair(ds.gb, p1.GB_PAIR, ctxs[i], sigs[i]))
        stage["run"], _ = bench(lambda i: ds.run(ctxs[i]))

        for k, v in stage.items():
            weighted[k][0] += v * n
            weighted[k][1] += n
        fold_rows.append({"fold": fd["fold"], "n": n, "ns_per_call": stage, "activations": local})

    pooled = {k: total / n for k, (total, n) in weighted.items()}
    run = pooled["run"]
    shares = {k: pooled[k] / run for k in ("hybrid", "gate", "sig", "gb")}
    rates = {k: (v / activations["n"] if k != "n" else v) for k, v in activations.items()}

    print(json.dumps({
        "phase": "p14_downstream_residual_profile",
        "cache_schema": state["schema_version"],
        "corpus_fingerprint": state["corpus_fingerprint"],
        "repeats": REPEATS,
        "pooled_ns_per_call": pooled,
        "isolated_share_vs_run": shares,
        "activation_counts": activations,
        "activation_rates": rates,
        "folds": fold_rows,
        "note": "Stage timings use cached upstream intermediates and are directional, not additive.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
