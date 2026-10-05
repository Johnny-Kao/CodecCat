from __future__ import annotations

import argparse
import json
import time

import joblib

import charset_candidate_calibration_ab as calmod
import charset_p2_multihop_kernel_tournament as p2
import charset_p7_specialist_microkernel_tournament as p7
import charset_p8_context_metadata_tournament as p8
import charset_p12_normalized_count_tournament as p12

TIMING_REPEATS = 31


class CalDownstream(p7.P7Downstream):
    def __init__(self, *args, mode="baseline", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self._cal_route = None
        self._cal_coef = None
        self._cal_pos = None
        if self.cal is not None and mode in ("route_local", "algebraic", "combined"):
            self._cal_route = {
                route: {cand: self.cal_static[(route, cand)] for cand in self.classes}
                for route in calmod.ROUTES
            }
        if self.cal is not None and mode in ("algebraic", "combined"):
            _, w, _ = self.cal
            v = w[0]
            self._cal_coef = float(v[0] + v[1] + v[2])
            self._cal_pos = (0.0, float(v[3]), float(2.0 * v[3]))

    def choose_cal(self, ctx):
        if self.mode == "baseline" or self.cal is None or len(ctx.rank) < 3:
            return super().choose_cal(ctx)

        _, w, b = self.cal
        v = w[0]
        scores = ctx.sorted_scores
        top = float(scores[0])
        second = float(scores[1])
        cand = ctx.rank[:calmod.TOP_CANDIDATES]

        if self.mode in ("common_hoist", "route_local"):
            common = float(b[0])
            common += v[4] * float(ctx.has_utf8_bom)
            common += v[5] * float(ctx.strict_utf8)
            common += v[6] * float(ctx.ascii_only)
            static = self.cal_static if self.mode == "common_hoist" else self._cal_route[ctx.route]

            def one(c, i):
                score = float(scores[i])
                z = common
                z += v[0] * score + v[1] * (score - top) + v[2] * (score - second) + v[3] * i
                z += static[(ctx.route, c)] if self.mode == "common_hoist" else static[c]
                return z

            z0 = one(cand[0], 0)
            z1 = one(cand[1], 1)
            z2 = one(cand[2], 2)
        else:
            common = float(b[0])
            common -= v[1] * top
            common -= v[2] * second
            common += v[4] * float(ctx.has_utf8_bom)
            common += v[5] * float(ctx.strict_utf8)
            common += v[6] * float(ctx.ascii_only)
            if self.mode == "algebraic":
                static0 = self.cal_static[(ctx.route, cand[0])]
                static1 = self.cal_static[(ctx.route, cand[1])]
                static2 = self.cal_static[(ctx.route, cand[2])]
            else:
                static = self._cal_route[ctx.route]
                static0, static1, static2 = static[cand[0]], static[cand[1]], static[cand[2]]
            coef = self._cal_coef
            pos = self._cal_pos
            z0 = common + coef * float(scores[0]) + pos[0] + static0
            z1 = common + coef * float(scores[1]) + pos[1] + static1
            z2 = common + coef * float(scores[2]) + pos[2] + static2

        winner = 0
        best = z0
        if z1 > best:
            winner = 1
            best = z1
        if z2 > best:
            winner = 2
        if winner == 0:
            return list(ctx.rank)
        chosen = cand[winner]
        out = list(ctx.rank)
        if chosen in out:
            out.remove(chosen)
            out.insert(0, chosen)
        return out


CANDIDATES = {
    "p12_baseline": "baseline",
    "common_hoist": "common_hoist",
    "route_local": "route_local",
    "algebraic": "algebraic",
    "combined": "combined",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()
    state = joblib.load(args.state)
    assert state["schema_version"] == 1
    assert sum(len(f["rows"]) for f in state["folds"]) == 418

    classes, families = state["classes"], state["families"]
    pooled = {n: {"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}
    folds = []

    for fd in state["folds"]:
        models, rows = fd["models"], fd["rows"]
        fused = {r: p2.fuse_scaler_linear(mt) for r, mt in models.items()}
        meta = {r: p8.RouteMeta(mt) for r, mt in models.items()}
        dsmap = {
            name: CalDownstream(
                fd["cal"], fd["triad_cal"], fd["sig_cal"], fd["gb_cal"],
                classes, families, models, True,
                scalar_triad=True, inline_pairs=False, mode=mode,
            )
            for name, mode in CANDIDATES.items()
        }

        ctxs = [
            p12.build_ctx(
                s, fused[s["route"]], meta[s["route"]],
                {"uni_mode":"assign_multiply","bi_mode":"assign_multiply"},
            )
            for s in rows
        ]
        refs = [dsmap["p12_baseline"].run(ctx) for ctx in ctxs]
        fs = {n: {"n":0,"hits":0,"mismatch":0,"elapsed":0,"calls":0} for n in CANDIDATES}

        for s, ctx, ref in zip(rows, ctxs, refs):
            for name, ds in dsmap.items():
                out = ds.run(ctx)
                st = fs[name]
                st["n"] += 1
                st["hits"] += int(out and out[0] == s["label"])
                st["mismatch"] += int(list(out) != list(ref))

        for name, ds in dsmap.items():
            if fs[name]["mismatch"]:
                continue
            sink = 0
            t0 = time.perf_counter_ns()
            for _ in range(TIMING_REPEATS):
                for ctx in ctxs:
                    out = ds.run(ctx)
                    sink += len(out)
            fs[name]["elapsed"] = time.perf_counter_ns() - t0
            fs[name]["calls"] = len(ctxs) * TIMING_REPEATS
            fs[name]["sink"] = sink

        fr = {"fold":fd["fold"],"n":len(rows),"candidates":{}}
        for name, st in fs.items():
            fr["candidates"][name] = {
                "top1": st["hits"]/max(1,st["n"]),
                "mismatch_n": st["mismatch"],
                "ns_per_call": st["elapsed"]/st["calls"] if st["calls"] else None,
            }
            for k in ("n","hits","mismatch","elapsed","calls"):
                pooled[name][k] += st[k]
        folds.append(fr)

    out = {}
    for name, st in pooled.items():
        ns = st["elapsed"]/st["calls"] if st["calls"] else None
        out[name] = {
            "n":st["n"],"hits":st["hits"],"top1":st["hits"]/max(1,st["n"]),
            "mismatch_n":st["mismatch"],"ns_per_call":ns,
        }
    base = out["p12_baseline"]["ns_per_call"]
    for row in out.values():
        if row["ns_per_call"] is not None:
            row["speedup_vs_p12_downstream"] = base / row["ns_per_call"]
            row["runtime_reduction_vs_p12_downstream"] = 1 - row["ns_per_call"] / base

    eligible = [(r["ns_per_call"], n) for n,r in out.items()
                if r["mismatch_n"] == 0 and r["hits"] == 365 and r["ns_per_call"] is not None]
    winner = min(eligible)[1] if eligible else None
    print(json.dumps({
        "phase":"p14_calibrator_hotpath_tournament",
        "canonical_target":{"n":418,"hits":365,"top1":365/418},
        "cache_schema":state["schema_version"],
        "corpus_fingerprint":state["corpus_fingerprint"],
        "timing_repeats":TIMING_REPEATS,
        "pooled":out,"winner":winner,"folds":folds,
        "decision_rule":"Preserve exact locked-P12 final ranking and 365/418; choose fastest same-run exact-equivalent downstream candidate."
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
