from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter

import joblib
import numpy as np

import charset_external_scorer_domain_shift_ab as base
import charset_external_scorer_learning_curve as lc
import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_post_gate_050_error_decomposition as gate050
import charset_canonical_guarded_final_validation as canon
from charset_cost_aware_routing_tree import collect as collect_legacy


OUTER_FOLDS = 4
TRAIN_N = 300
MODES = ("baseline_hmt768", "rl_hmt4096")


def hmt(data: bytes, total: int) -> bytes:
    if len(data) <= total:
        return data
    head = total // 3
    middle = total - 2 * head
    center = len(data) // 2
    left = center - middle // 2
    right = left + middle
    return data[:head] + data[left:right] + data[-head:]


def feature_from_sample(sampled: bytes) -> np.ndarray:
    a = np.frombuffer(sampled, dtype=np.uint8)
    n = max(1, len(a))
    unigram = np.bincount(a, minlength=256).astype(np.float32) / n
    bigram = np.zeros(256, dtype=np.float32)
    if len(a) >= 2:
        bins = a[:-1] + a[1:]
        bigram = np.bincount(bins, minlength=256).astype(np.float32) / (len(a) - 1)
    scalars = np.array([
        math.log2(n + 1) / 16.0,
        float(np.mean(a >= 128)) if len(a) else 0.0,
        float(np.mean(a == 0)) if len(a) else 0.0,
        float(np.mean((a >= 32) & (a <= 126))) if len(a) else 0.0,
        float(np.mean(a == 10)) if len(a) else 0.0,
        float(np.mean(a == 13)) if len(a) else 0.0,
    ], dtype=np.float32)
    return np.concatenate([unigram, bigram, scalars])


def make_feature(mode):
    def scorer(data: bytes) -> np.ndarray:
        if mode == "rl_hmt4096" and base.route_bucket(data) == "RL":
            return feature_from_sample(hmt(data, 4096))
        return feature_from_sample(hmt(data, 768))
    return scorer


def reconstruct_external(state):
    rows = []
    seen = set()
    for fd in state["folds"]:
        for s in fd["rows"]:
            key = (
                s["warc_path"],
                s.get("host", ""),
                s["route"],
                s["label"],
                hashlib.sha256(s["data"]).digest(),
            )
            if key not in seen:
                seen.add(key)
                rows.append(s)
    return rows


def raw_legacy_rows():
    rows, _ = collect_legacy()
    out = []
    for row in rows:
        data = row["path"].read_bytes()
        label = base.normalize_label(base.canonical_encoding(row["encoding"])) or row["encoding"]
        out.append({"data": data, "label": label, "route": base.route_bucket(data)})
    return out


def legacy_arrays(rows, scorer):
    X = np.stack([scorer(r["data"]) for r in rows])
    y = np.asarray([r["label"] for r in rows], dtype=object)
    b = np.asarray([r["route"] for r in rows], dtype=object)
    return X, y, b


def evaluate_mode(mode, external, fold_assign, legacy_rows):
    scorer = make_feature(mode)
    old = base.scorer_features
    base.scorer_features = scorer
    try:
        legacy_X, legacy_y, legacy_b = legacy_arrays(legacy_rows, scorer)
        classes, families = calmod.build_vocab(legacy_y)
        pooled = Counter()
        route_pooled = {r: Counter() for r in ("U", "N", "RL", "RH")}
        folds = []
        started = time.perf_counter()

        for fold in range(OUTER_FOLDS):
            train_pool = [s for s in external if fold_assign[s["warc_path"]] != fold]
            test_rows = [s for s in external if fold_assign[s["warc_path"]] == fold]
            ext_train = lc.deterministic_nested_subset(train_pool, TRAIN_N)

            tf = time.perf_counter()
            models = lc.fit_route_models(ext_train, legacy_X, legacy_y, legacy_b)
            cal = calmod.crossfit_calibration_rows(
                train_pool, legacy_X, legacy_y, legacy_b, classes, families
            )
            triad_cal = tri.crossfit_triad_rows(
                train_pool, legacy_X, legacy_y, legacy_b, classes, families
            )
            sig_cal = canon.fit_one(
                canon.SIG_PAIR, train_pool, legacy_X, legacy_y, legacy_b, classes, families
            )
            gb_cal = canon.fit_one(
                canon.GB_PAIR, train_pool, legacy_X, legacy_y, legacy_b, classes, families
            )

            c = Counter()
            routes = {r: Counter() for r in ("U", "N", "RL", "RH")}
            for s in test_rows:
                model = models.get(s["route"])
                if model is None or s["label"] not in model[1].classes_:
                    continue
                _, hybrid = tri.make_hybrid(model, cal, s, classes, families)
                gated = gate050.apply_gated(triad_cal, model, s, hybrid)
                sig = canon.apply_one(sig_cal, canon.SIG_PAIR, model, s, gated)
                out = canon.apply_guarded_gb(gb_cal, model, s, sig)
                hit = int(out and out[0] == s["label"])
                c["n"] += 1
                c["hits"] += hit
                routes[s["route"]]["n"] += 1
                routes[s["route"]]["hits"] += hit

            pooled.update(c)
            for r, v in routes.items():
                route_pooled[r].update(v)

            folds.append({
                "fold": fold,
                "n": c["n"],
                "hits": c["hits"],
                "top1": c["hits"] / max(1, c["n"]),
                "training_seconds": time.perf_counter() - tf,
            })

        return {
            "mode": mode,
            "n": pooled["n"],
            "hits": pooled["hits"],
            "top1": pooled["hits"] / max(1, pooled["n"]),
            "routes": {
                r: {
                    "n": v["n"],
                    "hits": v["hits"],
                    "top1": v["hits"] / max(1, v["n"]),
                }
                for r, v in route_pooled.items()
                if v["n"]
            },
            "folds": folds,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        base.scorer_features = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()

    state = joblib.load(args.state)
    external = reconstruct_external(state)
    legacy_rows = raw_legacy_rows()

    paths = sorted(set(s["warc_path"] for s in external))
    fold_assign = {
        p: int.from_bytes(
            hashlib.sha256(("learning-curve:" + p).encode()).digest()[:8], "big"
        ) % OUTER_FOLDS
        for p in paths
    }

    results = {
        mode: evaluate_mode(mode, external, fold_assign, legacy_rows)
        for mode in MODES
    }

    b = results["baseline_hmt768"]
    c = results["rl_hmt4096"]

    print(json.dumps({
        "phase": "r7_paired_fold_rebuild",
        "reconstructed_corpus": {
            "n": len(external),
            "cached_state_external_n": state["external_n"],
            "note": (
                "The cached state retains 418 evaluable rows from the original 420-row external corpus. "
                "Both arms are retrained on this identical 418-row reconstruction, so the A/B delta is controlled."
            ),
        },
        "results": results,
        "paired_delta": {
            "hits": c["hits"] - b["hits"],
            "top1_pp": (c["top1"] - b["top1"]) * 100.0,
            "rl_hits": c["routes"].get("RL", {}).get("hits", 0) - b["routes"].get("RL", {}).get("hits", 0),
            "rl_top1_pp": (
                c["routes"].get("RL", {}).get("top1", 0.0)
                - b["routes"].get("RL", {}).get("top1", 0.0)
            ) * 100.0,
        },
        "methodology": {
            "same_reconstructed_rows": True,
            "same_fold_assignment": True,
            "same_legacy_rows": True,
            "all_models_and_downstream_refit_per_arm": True,
            "only_difference": "RL HMT768 vs RL HMT4096",
            "cc_main_2026_30_used": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
