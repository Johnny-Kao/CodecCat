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

TIMING_REPEATS = 21


class RouteMeta:
    __slots__ = ("classes_tuple", "classes_array")

    def __init__(self, model_tuple):
        self.classes_array = model_tuple[1].classes_
        self.classes_tuple = tuple(self.classes_array)


class FastCtx:
    __slots__ = (
        "data", "route", "rank", "sorted_scores", "raw_scores", "classes",
        "has_utf8_bom", "strict_utf8", "ascii_only", "high_byte_ratio",
        "nul_ratio", "sample_len_4k", "replacement_rate",
    )

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw[name])


def build_ctx_fast(
    model_tuple, s, fused_model, *, meta=None, raw_direct=False,
    sample_view=False, slots_ctx=False,
):
    x = p2.features_combined(s["data"])
    raw = p2.fused_raw(fused_model, x)
    order = np.argsort(raw)[::-1]

    sample = memoryview(s["data"])[:4096] if sample_view else s["data"][:4096]
    arr = np.frombuffer(sample, dtype=np.uint8)
    strict = rr.strict_utf8(s["data"])
    bom = rr.has_utf8_bom(s["data"])

    if len(arr):
        high_count = int(np.count_nonzero(arr >= 128))
        nul_count = int(np.count_nonzero(arr == 0))
        high = high_count / len(arr)
        nul = nul_count / len(arr)
        ascii_flag = high_count == 0
    else:
        high = nul = 0.0
        ascii_flag = True

    if meta is None:
        classes_array = model_tuple[1].classes_
        classes_tuple = tuple(classes_array)
    else:
        classes_array = meta.classes_array
        classes_tuple = meta.classes_tuple

    if raw_direct:
        raw_scores = raw
        sorted_scores = raw[order]
    else:
        sorted_scores = np.asarray(raw)[order]
        raw_scores = np.asarray(raw)

    ctx_type = FastCtx if slots_ctx else p3.Ctx
    return ctx_type(
        data=s["data"],
        route=s["route"],
        rank=tuple(classes_array[order]),
        sorted_scores=sorted_scores,
        raw_scores=raw_scores,
        classes=classes_tuple,
        has_utf8_bom=bom,
        strict_utf8=strict,
        ascii_only=ascii_flag,
        high_byte_ratio=high,
        nul_ratio=nul,
        sample_len_4k=len(arr),
        replacement_rate=None,
    )


CANDIDATES = {
    "p7_baseline": dict(meta=False, raw=False, view=False, slots=False),
    "hop1_precomputed_meta": dict(meta=True, raw=False, view=False, slots=False),
    "hop1_raw_direct": dict(meta=False, raw=True, view=False, slots=False),
    "hop1_sample_view": dict(meta=False, raw=False, view=True, slots=False),
    "hop1_slots_ctx": dict(meta=False, raw=False, view=False, slots=True),
    "hop2_meta_raw": dict(meta=True, raw=True, view=False, slots=False),
    "hop2_view_slots": dict(meta=False, raw=False, view=True, slots=True),
    "hop3_all_safe": dict(meta=True, raw=True, view=True, slots=True),
}


def make_ctx(name, cfg, mt, s, fused_model, route_meta):
    if name == "p7_baseline":
        return p3.build_ctx(mt, s, fused_model, True)
    return build_ctx_fast(
        mt,
        s,
        fused_model,
        meta=route_meta if cfg["meta"] else None,
        raw_direct=cfg["raw"],
        sample_view=cfg["view"],
        slots_ctx=cfg["slots"],
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
        meta = {r: RouteMeta(mt) for r, mt in models.items()}
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
            ctx = p3.build_ctx(models[s["route"]], s, fused[s["route"]], True)
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
                out = ds.run(make_ctx(name, cfg, mt, s, fused_model, route_meta))
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
                            name,
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

    base_ns = out["p7_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p7"] = base_ns / row["ns_per_call"]
            row["runtime_reduction_vs_p7"] = 1 - row["ns_per_call"] / base_ns

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
                "phase": "p8_context_metadata_multihop_tournament",
                "canonical_target": {"n": 418, "hits": 365, "top1": 365 / 418},
                "cache_schema": state["schema_version"],
                "corpus_fingerprint": state["corpus_fingerprint"],
                "timing_repeats": TIMING_REPEATS,
                "pooled": out,
                "winner": winner,
                "folds": folds,
                "decision_rule": (
                    "Preserve exact P7 final ranking and 365/418; "
                    "choose fastest same-run exact-equivalent candidate."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
