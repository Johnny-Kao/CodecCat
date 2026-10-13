from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from codeccat import load_bundle
from codeccat.features import baseline_from_array, full_b_extra_from_array, hmt768_array
from codeccat.runtime import Runtime, _context, _signals, family

from h2_fused_candidate_mask_ab import collect, dedupe, normalize


FRACTIONS = (0.50, 0.75)


def policy_names():
    out = ["control"]
    for frac in FRACTIONS:
        tag = str(frac).replace(".", "p")
        out.extend([
            f"margin_{tag}",
            f"same_family_{tag}",
            f"different_family_{tag}",
        ])
    return out


def baseline_and_full(runtime: Runtime, data: bytes):
    bundle = runtime.bundle
    signals = _signals(data)
    array = hmt768_array(data)
    baseline_vector = baseline_from_array(array)

    bmodel = bundle.baseline.route_models[signals.route]
    bctx = _context(data, signals, bmodel, baseline_vector)
    baseline_rank = runtime.baseline.rank_context(bctx)
    baseline_final = tuple(runtime._apply_rl_specialist(bctx, baseline_rank))

    threshold = bundle.gate_thresholds.get(signals.route)
    margin = (
        float(bctx.sorted_scores[0] - bctx.sorted_scores[1])
        if len(bctx.sorted_scores) >= 2
        else float("inf")
    )
    current_escalate = (
        runtime.full_b is not None
        and threshold is not None
        and signals.route in ("U", "RH")
        and margin <= threshold
    )

    full_final = None
    if current_escalate:
        extra = full_b_extra_from_array(array, signals.route)
        full_vector = __import__("numpy").concatenate([baseline_vector, extra])
        fmodel = bundle.full_b.route_models[signals.route]
        fctx = _context(data, signals, fmodel, full_vector)
        full_final = tuple(runtime.full_b.rank_context(fctx))

    top1 = bctx.rank[0] if bctx.rank else None
    top2 = bctx.rank[1] if len(bctx.rank) > 1 else None
    fam1 = family(top1)
    fam2 = family(top2)
    return {
        "signals": signals,
        "margin": margin,
        "threshold": threshold,
        "current_escalate": current_escalate,
        "baseline_final": baseline_final,
        "full_final": full_final,
        "top1": top1,
        "top2": top2,
        "family1": fam1,
        "family2": fam2,
        "same_family": fam1 == fam2,
    }


def should_escalate(info, policy: str) -> bool:
    if not info["current_escalate"]:
        return False
    if policy == "control":
        return True

    kind, frac_tag = policy.rsplit("_", 1)
    frac = float(frac_tag.replace("p", "."))
    threshold = info["threshold"]
    if threshold is None:
        return False

    strong_enough = info["margin"] > frac * float(threshold)
    if not strong_enough:
        return True

    if kind == "margin":
        return False
    if kind == "same_family":
        return not info["same_family"]
    if kind == "different_family":
        return info["same_family"]
    raise ValueError(policy)


def chosen_rank(info, policy):
    if should_escalate(info, policy):
        assert info["full_final"] is not None
        return info["full_final"]
    return info["baseline_final"]


def taxonomy(rows, runtime):
    c = Counter()
    by_route = defaultdict(Counter)
    by_family_relation = defaultdict(Counter)
    details = []

    for s in rows:
        truth = normalize(s["label"])
        info = baseline_and_full(runtime, s["data"])
        if not info["current_escalate"]:
            continue

        b = info["baseline_final"][0] if info["baseline_final"] else None
        f = info["full_final"][0] if info["full_final"] else None
        bc = b == truth
        fc = f == truth

        c["escalated"] += 1
        c["changed_top1"] += int(b != f)
        c["unchanged_top1"] += int(b == f)

        if (not bc) and fc:
            cls = "beneficial"
        elif bc and (not fc):
            cls = "harmful"
        elif (not bc) and (not fc) and b != f:
            cls = "wrong_to_different_wrong"
        elif bc and fc:
            cls = "correct_to_correct"
        else:
            cls = "wrong_unchanged"

        c[cls] += 1
        by_route[info["signals"].route][cls] += 1
        rel = "same_family" if info["same_family"] else "different_family"
        by_family_relation[rel][cls] += 1

        details.append({
            "route": info["signals"].route,
            "margin": info["margin"],
            "threshold": info["threshold"],
            "top1": info["top1"],
            "top2": info["top2"],
            "family1": info["family1"],
            "family2": info["family2"],
            "same_family": info["same_family"],
            "class": cls,
        })

    def margin_stats(group):
        vals = [x["margin"] for x in details if x["class"] == group]
        return {
            "n": len(vals),
            "p10": statistics.quantiles(vals, n=10, method="inclusive")[0] if len(vals) >= 2 else (vals[0] if vals else None),
            "p25": __import__("numpy").quantile(vals, .25).item() if vals else None,
            "p50": statistics.median(vals) if vals else None,
            "p75": __import__("numpy").quantile(vals, .75).item() if vals else None,
            "p90": statistics.quantiles(vals, n=10, method="inclusive")[8] if len(vals) >= 2 else (vals[0] if vals else None),
        }

    return {
        "counts": dict(c),
        "by_route": {k: dict(v) for k, v in by_route.items()},
        "by_family_relation": {k: dict(v) for k, v in by_family_relation.items()},
        "margin_by_class": {
            k: margin_stats(k)
            for k in ("beneficial", "harmful", "wrong_to_different_wrong", "correct_to_correct", "wrong_unchanged")
        },
    }


def evaluate_policy(rows, runtime, policy):
    c = Counter()
    routes = defaultdict(Counter)

    for s in rows:
        truth = normalize(s["label"])
        info = baseline_and_full(runtime, s["data"])
        control_rank = chosen_rank(info, "control")
        rank = chosen_rank(info, policy)

        control_hit = int(bool(control_rank) and control_rank[0] == truth)
        hit = int(bool(rank) and rank[0] == truth)

        c["n"] += 1
        c["control_hits"] += control_hit
        c["hits"] += hit
        c["control_triggers"] += int(info["current_escalate"])
        c["triggers"] += int(should_escalate(info, policy))

        if info["current_escalate"] and not should_escalate(info, policy):
            c["skipped_full_b"] += 1
            b = info["baseline_final"][0] if info["baseline_final"] else None
            f = info["full_final"][0] if info["full_final"] else None
            bc = b == truth
            fc = f == truth
            if (not bc) and fc:
                c["skipped_beneficial_full_b"] += 1
            if bc and (not fc):
                c["skipped_harmful_full_b"] += 1
            if b == f:
                c["skipped_noop_full_b"] += 1

        route = info["signals"].route
        rr = routes[route]
        rr["n"] += 1
        rr["control_hits"] += control_hit
        rr["hits"] += hit
        rr["control_triggers"] += int(info["current_escalate"])
        rr["triggers"] += int(should_escalate(info, policy))

    return {
        "n": c["n"],
        "hits": c["hits"],
        "accuracy": c["hits"] / max(1, c["n"]),
        "control_hits": c["control_hits"],
        "delta_hits": c["hits"] - c["control_hits"],
        "control_triggers": c["control_triggers"],
        "triggers": c["triggers"],
        "trigger_reduction_n": c["control_triggers"] - c["triggers"],
        "trigger_reduction_fraction": (c["control_triggers"] - c["triggers"]) / max(1, c["control_triggers"]),
        "skipped_full_b": c["skipped_full_b"],
        "skipped_beneficial_full_b": c["skipped_beneficial_full_b"],
        "skipped_harmful_full_b": c["skipped_harmful_full_b"],
        "skipped_noop_full_b": c["skipped_noop_full_b"],
        "routes": {
            r: {
                "n": x["n"],
                "delta_hits": x["hits"] - x["control_hits"],
                "control_triggers": x["control_triggers"],
                "triggers": x["triggers"],
                "trigger_reduction_n": x["control_triggers"] - x["triggers"],
            }
            for r, x in routes.items()
        },
    }


def rank_policy(runtime: Runtime, data: bytes, policy: str):
    bundle = runtime.bundle
    signals = _signals(data)
    array = hmt768_array(data)
    baseline_vector = baseline_from_array(array)

    bmodel = bundle.baseline.route_models[signals.route]
    bctx = _context(data, signals, bmodel, baseline_vector)
    baseline = runtime.baseline.rank_context(bctx)

    threshold = bundle.gate_thresholds.get(signals.route)
    margin = (
        float(bctx.sorted_scores[0] - bctx.sorted_scores[1])
        if len(bctx.sorted_scores) >= 2
        else float("inf")
    )
    current_escalate = (
        runtime.full_b is not None
        and threshold is not None
        and signals.route in ("U", "RH")
        and margin <= threshold
    )
    if current_escalate:
        top1 = bctx.rank[0] if bctx.rank else None
        top2 = bctx.rank[1] if len(bctx.rank) > 1 else None
        info = {
            "current_escalate": True,
            "margin": margin,
            "threshold": threshold,
            "same_family": family(top1) == family(top2),
        }
        if should_escalate(info, policy):
            extra = full_b_extra_from_array(array, signals.route)
            full_vector = __import__("numpy").concatenate([baseline_vector, extra])
            fmodel = bundle.full_b.route_models[signals.route]
            fctx = _context(data, signals, fmodel, full_vector)
            return tuple(runtime.full_b.rank_context(fctx))

    return tuple(runtime._apply_rl_specialist(bctx, baseline))


def benchmark(rows, runtime, policies, repeats=15):
    data = [s["data"] for s in rows]
    out = {}
    for policy in policies:
        vals = []
        sink = 0
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            for d in data:
                sink += len(rank_policy(runtime, d, policy))
            vals.append((time.perf_counter_ns() - t0) / max(1, len(data)))
        out[policy] = {
            "median_ns_per_sample": statistics.median(vals),
            "min_ns_per_sample": min(vals),
            "max_ns_per_sample": max(vals),
            "sink": sink,
        }

    control = out["control"]["median_ns_per_sample"]
    for policy in policies:
        out[policy]["delta_ns_vs_control"] = out[policy]["median_ns_per_sample"] - control
        out[policy]["delta_fraction_vs_control"] = out[policy]["median_ns_per_sample"] / control - 1.0
    return {"repeats": repeats, "policies": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--expected-model-sha256", required=True)
    args = ap.parse_args()

    path = Path(args.model)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != args.expected_model_sha256:
        raise RuntimeError(f"model SHA mismatch: {actual}")

    runtime = Runtime(load_bundle(path))
    policies = policy_names()
    all_rows = []
    per_crawl = {}
    collection = {}

    for crawl in ("CC-MAIN-2026-39", "CC-MAIN-2026-34"):
        rows, paths, stats = collect(crawl)
        rows = dedupe(rows)
        all_rows.extend(rows)
        per_crawl[crawl] = {
            "taxonomy": taxonomy(rows, runtime),
            "policies": {p: evaluate_policy(rows, runtime, p) for p in policies},
        }
        collection[crawl] = {"n": len(rows), "warc_paths_considered": len(paths), "stats": stats}

    pooled = dedupe(all_rows)
    pooled_policies = {p: evaluate_policy(pooled, runtime, p) for p in policies}
    timing = benchmark(pooled, runtime, policies)

    decisions = {}
    for p in policies:
        if p == "control":
            continue
        cross_nonnegative = all(per_crawl[c]["policies"][p]["delta_hits"] >= 0 for c in per_crawl)
        pooled_nonnegative = pooled_policies[p]["delta_hits"] >= 0
        trigger_reduced = pooled_policies[p]["trigger_reduction_n"] > 0
        runtime_improved = timing["policies"][p]["delta_ns_vs_control"] < 0
        decisions[p] = {
            "cross_crawl_nonnegative": cross_nonnegative,
            "pooled_nonnegative": pooled_nonnegative,
            "trigger_reduced": trigger_reduced,
            "runtime_improved": runtime_improved,
            "passes_all": cross_nonnegative and pooled_nonnegative and trigger_reduced and runtime_improved,
        }

    print(json.dumps({
        "phase": "h3_comprehensive_full_b_necessity_study",
        "policy": {
            "r22_used": False,
            "parameters_tuned": False,
            "retrained": False,
            "release_candidate_modified": False,
            "fractions": FRACTIONS,
            "candidate_families": ["margin", "same_family", "different_family"],
            "broad_sweep": False,
        },
        "collection": collection,
        "per_crawl": per_crawl,
        "pooled": {
            "taxonomy": taxonomy(pooled, runtime),
            "policies": pooled_policies,
        },
        "timing": timing,
        "decision": decisions,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
