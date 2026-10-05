from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np

import charset_minimal_reranker_ab as rr
import charset_p2_multihop_kernel_tournament as p2
import charset_p3_allocation_lazy_tournament as p3
import charset_p7_specialist_microkernel_tournament as p7
import charset_p8_context_metadata_tournament as p8

TIMING_REPEATS = 23


def build_ctx(
    model_tuple,
    s,
    fused_model,
    meta,
    *,
    rank_gather=False,
    top3_scores=False,
    argsort_method=False,
    byte_bincount=False,
):
    x = p2.features_combined(s["data"])
    raw = p2.fused_raw(fused_model, x)
    order = raw.argsort()[::-1] if argsort_method else np.argsort(raw)[::-1]

    sample = s["data"][:4096]
    arr = np.frombuffer(sample, dtype=np.uint8)
    strict = rr.strict_utf8(s["data"])
    bom = rr.has_utf8_bom(s["data"])

    if len(arr):
        if byte_bincount:
            counts = np.bincount(arr, minlength=256)
            high_count = int(counts[128:].sum())
            nul_count = int(counts[0])
        else:
            high_count = int(np.count_nonzero(arr >= 128))
            nul_count = int(np.count_nonzero(arr == 0))
        high = high_count / len(arr)
        nul = nul_count / len(arr)
        ascii_flag = high_count == 0
    else:
        high = nul = 0.0
        ascii_flag = True

    if rank_gather:
        rank = tuple(meta.classes_tuple[int(i)] for i in order)
    else:
        rank = tuple(meta.classes_array[order])

    score_order = order[:3] if top3_scores else order
    sorted_scores = raw[score_order]

    return p3.Ctx(
        data=s["data"],
        route=s["route"],
        rank=rank,
        sorted_scores=sorted_scores,
        raw_scores=raw,
        classes=meta.classes_tuple,
        has_utf8_bom=bom,
        strict_utf8=strict,
        ascii_only=ascii_flag,
        high_byte_ratio=high,
        nul_ratio=nul,
        sample_len_4k=len(arr),
        replacement_rate=None,
    )


CANDIDATES = {
    "p8_baseline": dict(rank=False, top3=False, method=False, bincount=False),
    "hop1_rank_gather": dict(rank=True, top3=False, method=False, bincount=False),
    "hop1_top3_scores": dict(rank=False, top3=True, method=False, bincount=False),
    "hop1_argsort_method": dict(rank=False, top3=False, method=True, bincount=False),
    "hop1_byte_bincount": dict(rank=False, top3=False, method=False, bincount=True),
    "hop2_method_top3": dict(rank=False, top3=True, method=True, bincount=False),
    "hop2_rank_top3": dict(rank=True, top3=True, method=False, bincount=False),
    "hop2_rank_top3_method": dict(rank=True, top3=True, method=True, bincount=False),
    "hop3_all": dict(rank=True, top3=True, method=True, bincount=True),
}


def make_ctx(cfg, mt, s, fused_model, meta):
    return build_ctx(
        mt,
        s,
        fused_model,
        meta,
        rank_gather=cfg["rank"],
        top3_scores=cfg["top3"],
        argsort_method=cfg["method"],
        byte_bincount=cfg["bincount"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()

    state = joblib.load(args.state)
    assert state["schema_version"] == 1
    assert sum(len(f["rows"]) for f in state["folds"]) == 418

    classes = state["classes"]
    families = state["families"]
    pooled = {
        n: {"n": 0, "hits": 0, "mismatch": 0, "elapsed": 0, "calls": 0}
        for n in CANDIDATES
    }
    folds = []

    for fd in state["folds"]:
        models = fd["models"]
        rows = fd["rows"]
        fused = {r: p2.fuse_scaler_linear(mt) for r, mt in models.items()}
        meta = {r: p8.RouteMeta(mt) for r, mt in models.items()}

        ds = p7.P7Downstream(
            fd["cal"],
            fd["triad_cal"],
            fd["sig_cal"],
            fd["gb_cal"],
            classes,
            families,
            models,
            True,
            scalar_triad=True,
            inline_pairs=False,
        )

        refs = []
        for s in rows:
            mt = models[s["route"]]
            ctx = build_ctx(
                mt,
                s,
                fused[s["route"]],
                meta[s["route"]],
            )
            refs.append(ds.run(ctx))

        fs = {
            n: {"n": 0, "hits": 0, "mismatch": 0, "elapsed": 0, "calls": 0}
            for n in CANDIDATES
        }

        for s, ref in zip(rows, refs):
            mt = models[s["route"]]
            route_meta = meta[s["route"]]
            fused_model = fused[s["route"]]
            for name, cfg in CANDIDATES.items():
                out = ds.run(make_ctx(cfg, mt, s, fused_model, route_meta))
                st = fs[name]
                st["n"] += 1
                st["hits"] += int(out and out[0] == s["label"])
                st["mismatch"] += int(list(out) != list(ref))

        for name, cfg in CANDIDATES.items():
            if fs[name]["mismatch"]:
                continue
            sink = 0
            t0 = time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for s in rows:
                    mt = models[s["route"]]
                    out = ds.run(
                        make_ctx(
                            cfg,
                            mt,
                            s,
                            fused[s["route"]],
                            meta[s["route"]],
                        )
                    )
                    sink += len(out)
            dt = time.perf_counter_ns() - t0
            fs[name]["elapsed"] = dt
            fs[name]["calls"] = len(rows) * TIMING_REPEATS
            fs[name]["sink"] = sink

        fr = {"fold": fd["fold"], "n": len(rows), "candidates": {}}
        for name, st in fs.items():
            fr["candidates"][name] = {
                "top1": st["hits"] / max(1, st["n"]),
                "mismatch_n": st["mismatch"],
                "ns_per_call": st["elapsed"] / st["calls"] if st["calls"] else None,
            }
            for k in ("n", "hits", "mismatch", "elapsed", "calls"):
                pooled[name][k] += st[k]
        folds.append(fr)

    out = {}
    for name, st in pooled.items():
        ns = st["elapsed"] / st["calls"] if st["calls"] else None
        out[name] = {
            "n": st["n"],
            "hits": st["hits"],
            "top1": st["hits"] / max(1, st["n"]),
            "mismatch_n": st["mismatch"],
            "ns_per_call": ns,
        }

    base_ns = out["p8_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p8"] = base_ns / row["ns_per_call"]
            row["runtime_reduction_vs_p8"] = 1 - row["ns_per_call"] / base_ns

    eligible = [
        (r["ns_per_call"], n)
        for n, r in out.items()
        if r["mismatch_n"] == 0
        and r["hits"] == 365
        and r["ns_per_call"] is not None
    ]
    winner = min(eligible)[1] if eligible else None

    print(
        json.dumps(
            {
                "phase": "p9_ordering_rank_multihop_tournament",
                "canonical_target": {"n": 418, "hits": 365, "top1": 365 / 418},
                "cache_schema": state["schema_version"],
                "corpus_fingerprint": state["corpus_fingerprint"],
                "timing_repeats": TIMING_REPEATS,
                "pooled": out,
                "winner": winner,
                "folds": folds,
                "decision_rule": (
                    "Preserve exact P8 final ranking and 365/418; "
                    "choose fastest same-run exact-equivalent candidate."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
