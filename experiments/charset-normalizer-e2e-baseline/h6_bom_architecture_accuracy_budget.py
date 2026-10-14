from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import chardet
import numpy as np
from charset_normalizer import from_bytes

from codeccat import load_bundle
from codeccat.features import baseline_from_array, full_b_extra_from_array, hmt768_array
import codeccat.runtime as runtime_mod
from codeccat.runtime import Runtime, _context, _signals, family
from h2_fused_candidate_mask_ab import collect, dedupe, normalize


DEVELOPMENT = ("CC-MAIN-2026-39", "CC-MAIN-2026-34", "CC-MAIN-2026-21", "CC-MAIN-2026-17")
NEW_EVALUATION = ("CC-MAIN-2026-12", "CC-MAIN-2026-08")
POLICIES = (
    "control",
    "final_bom_only",
    "defer_bom_rule",
    "unified_bom",
    "full_b_always",
    "unified_bom_always",
)
PAIRED = {"utf-8", "utf-8-sig"}
ORIGINAL_RULE = runtime_mod._rule_rerank


def _deferred_rule(ctx):
    # Remove only the first, unconditional BOM override.
    # Retain every other original downstream rule and model as-is.
    if ctx.has_utf8_bom:
        return ORIGINAL_RULE(replace(ctx, has_utf8_bom=False))
    return ORIGINAL_RULE(ctx)


def _final_owner(data, rank):
    if not rank or rank[0] not in PAIRED:
        return tuple(rank)
    desired = "utf-8-sig" if data.startswith(bytes((0xEF, 0xBB, 0xBF))) else "utf-8"
    if desired not in rank or rank[0] == desired:
        return tuple(rank)
    return (desired,) + tuple(x for x in rank if x != desired)


def _full_b_always(rt, data):
    sig = _signals(data)
    if rt.full_b is None or sig.route not in ("U", "RH"):
        return rt.rank(data)
    arr = hmt768_array(data)
    vec = baseline_from_array(arr)
    extra = full_b_extra_from_array(arr, sig.route)
    full_vec = np.concatenate((vec, extra))
    model = rt.bundle.full_b.route_models[sig.route]
    ctx = _context(data, sig, model, full_vec)
    return tuple(rt.full_b.rank_context(ctx))


def predict(rt, data, policy):
    if policy == "control":
        return rt.rank(data)

    defer = policy in {"defer_bom_rule", "unified_bom", "unified_bom_always"}
    full_b = policy in {"full_b_always", "unified_bom_always"}
    final_owner = policy in {"final_bom_only", "unified_bom", "unified_bom_always"}

    context = patch.object(runtime_mod, "_rule_rerank", _deferred_rule) if defer else nullcontext()
    with context:
        rank = _full_b_always(rt, data) if full_b else rt.rank(data)
    return _final_owner(data, rank) if final_owner else tuple(rank)


def digest(data):
    return hashlib.sha256(data).digest()


def preflight(rt):
    samples = (
        b"hello",
        b"A" * 512 + bytes([0x80]),
        bytes([0x80, 0x81]) * 300,
        bytes([0, 255]) * 300,
        bytes([0xEF, 0xBB, 0xBF]) + b"hello",
    )
    assert [_signals(x).route for x in samples] == ["U", "RL", "RH", "N", "U"]
    for data in samples:
        control = rt.rank(data)
        assert control
        for policy in POLICIES:
            pred = predict(rt, data, policy)
            assert pred, (policy, "empty rank")
            assert set(pred) == set(control), (policy, "rank membership changed")
            assert len(pred) == len(control), (policy, "rank length changed")
    print("H6_PREFLIGHT_PASS", flush=True)


def evaluate(rows, rt):
    counts = defaultdict(Counter)
    routes = defaultdict(lambda: defaultdict(Counter))
    families = defaultdict(lambda: defaultdict(Counter))
    byte_labels = Counter()
    for sample in rows:
        data = sample["data"]
        truth = normalize(sample["label"])
        r = _signals(data).route
        fam = family(truth)
        bom = data.startswith(bytes((0xEF, 0xBB, 0xBF)))
        if bom:
            byte_labels["utf8_bom_bytes"] += 1
            byte_labels["utf8_bom_label_" + str(truth)] += 1
            if truth == "utf-8":
                byte_labels["bom_truth_utf8_not_sig"] += 1

        out = {policy: predict(rt, data, policy)[0] for policy in POLICIES}
        predicted_chardet = chardet.detect(data).get("encoding")
        out["chardet7"] = normalize(predicted_chardet) if predicted_chardet else None
        candidate = from_bytes(data).best()
        out["charset_normalizer"] = normalize(candidate.encoding) if candidate is not None else None

        control_good = out["control"] == truth
        for policy, pred in out.items():
            hit = pred == truth
            c = counts[policy]
            c["n"] += 1
            c["hits"] += int(hit)
            c["utf8_family_agreement"] += int(pred in PAIRED and truth in PAIRED)
            q = routes[r][policy]
            q["n"] += 1
            q["hits"] += int(hit)
            f = families[fam][policy]
            f["n"] += 1
            f["hits"] += int(hit)
            if policy != "control":
                changed = pred != out["control"]
                c["changed"] += int(changed)
                c["beneficial"] += int(hit and not control_good)
                c["harmful"] += int(control_good and not hit)
                q["beneficial"] += int(hit and not control_good)
                q["harmful"] += int(control_good and not hit)
                f["beneficial"] += int(hit and not control_good)
                f["harmful"] += int(control_good and not hit)
    def summarise(c):
        return {**dict(c), "accuracy": c["hits"] / max(1, c["n"])}
    return {
        "models": {key: summarise(v) for key, v in counts.items()},
        "routes": {r: {p: summarise(c) for p, c in inner.items()} for r, inner in routes.items()},
        "truth_families": {f: {p: summarise(c) for p, c in inner.items()} for f, inner in families.items()},
        "bom_label_disagreements": dict(byte_labels),
    }


def benchmark(rows, rt, repeats=9):
    samples = [s["data"] for s in rows]
    values = {p: [] for p in POLICIES}
    for rep in range(repeats):
        order = POLICIES if rep % 2 == 0 else tuple(reversed(POLICIES))
        for policy in order:
            t0 = time.perf_counter_ns()
            sink = 0
            for data in samples:
                sink += len(predict(rt, data, policy))
            elapsed = (time.perf_counter_ns() - t0) / max(1, len(samples))
            values[policy].append(elapsed)
            assert sink > 0
    median_control = statistics.median(values["control"])
    return {
        "repeats": repeats,
        "n": len(samples),
        "policies": {
            p: {
                "median_ns_per_sample": statistics.median(vals),
                "min_ns_per_sample": min(vals),
                "max_ns_per_sample": max(vals),
                "delta_fraction_vs_control": statistics.median(vals) / median_control - 1.0,
            }
            for p, vals in values.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sha256", required=True)
    a = ap.parse_args()

    path = Path(a.model)
    if hashlib.sha256(path.read_bytes()).hexdigest() != a.sha256:
        raise RuntimeError("frozen R21 checksum mismatch")
    rt = Runtime(load_bundle(path))
    preflight(rt)

    raw = {}
    meta = {}
    for crawl in DEVELOPMENT + NEW_EVALUATION:
        rows, paths, stats = collect(crawl)
        rows = dedupe(rows)
        raw[crawl] = rows
        meta[crawl] = {
            "before_overlap_filter": len(rows),
            "warc_paths_considered": len(paths),
            "collection_stats": stats,
        }

    all_known_hashes = {digest(s["data"]) for c in DEVELOPMENT for s in raw[c]}
    evaluation_hashes = set(all_known_hashes)
    dev = dedupe([s for c in DEVELOPMENT for s in raw[c]])
    fresh = []
    fresh_per = {}
    for crawl in NEW_EVALUATION:
        filtered = []
        for s in raw[crawl]:
            key = digest(s["data"])
            if key in evaluation_hashes:
                continue
            evaluation_hashes.add(key)
            filtered.append(s)
        meta[crawl]["after_exact_byte_overlap_filter"] = len(filtered)
        fresh.extend(filtered)
        fresh_per[crawl] = evaluate(filtered, rt)

    # Preserve original truth annotations, and exclude repeat bytes across fresh crawls.
    fresh = dedupe(fresh)
    dev_metrics = evaluate(dev, rt)
    fresh_metrics = evaluate(fresh, rt)
    timing = benchmark(fresh, rt)

    gates = {}
    for p in POLICIES[1:]:
        dev_delta = dev_metrics["models"][p]["hits"] - dev_metrics["models"]["control"]["hits"]
        new_delta = fresh_metrics["models"][p]["hits"] - fresh_metrics["models"]["control"]["hits"]
        fresh_nonnegative_each = all(
            fresh_per[c]["models"][p]["hits"] >= fresh_per[c]["models"]["control"]["hits"]
            for c in NEW_EVALUATION
        )
        gates[p] = {
            "development_delta_hits": dev_delta,
            "fresh_delta_hits": new_delta,
            "fresh_nonnegative_each": fresh_nonnegative_each,
            "fresh_no_regression": fresh_nonnegative_each and new_delta >= 0,
            "speed_delta_fraction": timing["policies"][p]["delta_fraction_vs_control"],
            "research_only_no_promotion": True,
        }

    result = {
        "phase": "h6_bom_architecture_accuracy_budget",
        "policy": {
            "model_retrained": False,
            "thresholds_retuned": False,
            "release_edited": False,
            "r22_used": False,
            "cc30_used": False,
            "new_crawls": NEW_EVALUATION,
            "source_labels_preserved": True,
            "exact_match_is_primary": True,
            "utf8_family_agreement_is_diagnostic_only": True,
            "policies_preregistered": POLICIES,
        },
        "collection": meta,
        "development": dev_metrics,
        "new_evaluation": fresh_metrics,
        "per_new_crawl": fresh_per,
        "timing_new_evaluation": timing,
        "candidate_gates": gates,
    }
    print("H6_RESULT_START", flush=True)
    print(json.dumps(result, sort_keys=True, indent=2))
    print("H6_RESULT_END", flush=True)


if __name__ == "__main__":
    main()
