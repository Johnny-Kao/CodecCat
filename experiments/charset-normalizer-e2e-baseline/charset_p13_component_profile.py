from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np

import charset_p2_multihop_kernel_tournament as p2
import charset_p7_specialist_microkernel_tournament as p7
import charset_p8_context_metadata_tournament as p8
import charset_p12_slice_hmt_tournament as p12

TIMING_REPEATS = 31
LOCKED_CFG = {"hmt": "bytes", "sample_decode": True, "inline_bom": True}


def time_loop(fn, items, repeats):
    sink = 0
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        for item in items:
            sink += fn(item)
    dt = time.perf_counter_ns() - t0
    return dt / max(1, len(items) * repeats), sink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()

    state = joblib.load(args.state)
    assert state["schema_version"] == 1
    assert sum(len(f["rows"]) for f in state["folds"]) == 418

    classes, families = state["classes"], state["families"]
    totals = {
        "feature": [0.0, 0],
        "linear_order": [0.0, 0],
        "sample_checks": [0.0, 0],
        "downstream": [0.0, 0],
        "full": [0.0, 0],
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

        # Precompute component inputs outside the component being timed.
        xs = [p12.features_locked(s["data"], "bytes") for s in rows]
        ctxs = [
            p12.build_ctx(s, fused[s["route"]], meta[s["route"]], LOCKED_CFG)
            for s in rows
        ]

        def feature_fn(s):
            x = p12.features_locked(s["data"], "bytes")
            return int(x.shape[0])

        feature_ns, sink1 = time_loop(feature_fn, rows, TIMING_REPEATS)

        indexed = list(zip(rows, xs))
        def linear_fn(pair):
            s, x = pair
            raw = p2.fused_raw(fused[s["route"]], x)
            order = raw.argsort()[::-1]
            rank = tuple(meta[s["route"]].classes_array[order])
            sorted_scores = raw[order]
            return len(rank) + len(sorted_scores)

        linear_ns, sink2 = time_loop(linear_fn, indexed, TIMING_REPEATS)

        def checks_fn(s):
            sample = s["data"][:4096]
            arr = np.frombuffer(sample, dtype=np.uint8)
            strict = p12.strict_from_sample(sample)
            bom = sample.startswith(p12.UTF8_BOM)
            if len(arr):
                high_count = int(np.count_nonzero(arr >= 128))
                nul_count = int(np.count_nonzero(arr == 0))
            else:
                high_count = nul_count = 0
            return int(strict) + int(bom) + high_count + nul_count + len(arr)

        checks_ns, sink3 = time_loop(checks_fn, rows, TIMING_REPEATS)

        def downstream_fn(ctx):
            return len(ds.run(ctx))

        downstream_ns, sink4 = time_loop(downstream_fn, ctxs, TIMING_REPEATS)

        def full_fn(s):
            ctx = p12.build_ctx(s, fused[s["route"]], meta[s["route"]], LOCKED_CFG)
            return len(ds.run(ctx))

        full_ns, sink5 = time_loop(full_fn, rows, TIMING_REPEATS)

        assert sink4 == sink5
        row = {
            "fold": fd["fold"],
            "n": len(rows),
            "ns_per_call": {
                "feature": feature_ns,
                "linear_order": linear_ns,
                "sample_checks": checks_ns,
                "downstream": downstream_ns,
                "full": full_ns,
            },
        }
        folds.append(row)

        n = len(rows)
        for k, v in row["ns_per_call"].items():
            totals[k][0] += v * n
            totals[k][1] += n

    pooled = {k: v[0] / max(1, v[1]) for k, v in totals.items()}
    component_sum = sum(pooled[k] for k in ("feature", "linear_order", "sample_checks", "downstream"))
    shares = {
        k: pooled[k] / component_sum
        for k in ("feature", "linear_order", "sample_checks", "downstream")
    }

    print(json.dumps({
        "phase": "p13_locked_p12_component_profile",
        "cache_schema": state["schema_version"],
        "corpus_fingerprint": state["corpus_fingerprint"],
        "timing_repeats": TIMING_REPEATS,
        "pooled_ns_per_call": pooled,
        "component_sum_ns": component_sum,
        "component_share_of_measured_sum": shares,
        "full_vs_component_sum_ratio": pooled["full"] / component_sum,
        "folds": folds,
        "interpretation": (
            "Component timings use precomputed inputs at boundaries and are for hotspot ranking; "
            "the full timing is authoritative for end-to-end performance."
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
