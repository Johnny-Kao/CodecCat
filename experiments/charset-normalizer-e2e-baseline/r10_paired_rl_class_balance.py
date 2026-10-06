from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import charset_external_scorer_domain_shift_ab as base
import charset_external_scorer_learning_curve as lc
import charset_candidate_calibration_ab as calmod
import charset_triad_specialist_ab as tri
import charset_post_gate_050_error_decomposition as gate050
import charset_canonical_guarded_final_validation as canon
from charset_cost_aware_routing_tree import collect as collect_legacy

OUTER_FOLDS = 4
TRAIN_N = 300
MODES = ("baseline", "rl_sqrt_balanced", "rl_balanced")


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


def legacy_arrays(rows):
    X = np.stack([base.scorer_features(r["data"]) for r in rows])
    y = np.asarray([r["label"] for r in rows], dtype=object)
    b = np.asarray([r["route"] for r in rows], dtype=object)
    return X, y, b


def fit_linear_weighted(X, y, mode, C=0.5):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if mode == "baseline":
        model = LogisticRegression(max_iter=2500, solver="lbfgs", C=C)
        model.fit(Xs, y)
        return scaler, model

    counts = Counter(y.tolist())
    n = len(y)
    k = len(counts)
    balanced = {label: n / (k * count) for label, count in counts.items()}

    if mode == "rl_balanced":
        sample_weight = np.asarray([balanced[label] for label in y], dtype=np.float64)
    elif mode == "rl_sqrt_balanced":
        sample_weight = np.asarray(
            [math.sqrt(balanced[label]) for label in y], dtype=np.float64
        )
    else:
        raise ValueError(mode)

    # Preserve average effective sample weight near 1 to avoid changing C scale.
    sample_weight *= len(sample_weight) / sample_weight.sum()
    model = LogisticRegression(max_iter=2500, solver="lbfgs", C=C)
    model.fit(Xs, y, sample_weight=sample_weight)
    return scaler, model


def make_fit_route_models(mode):
    def fit_route_models(ext_rows, legacy_X, legacy_y, legacy_b):
        models = {}
        for route in ("U", "N", "RL", "RH"):
            li = np.where(legacy_b == route)[0]
            Xtr = legacy_X[li]
            ytr = legacy_y[li]

            er = [s for s in ext_rows if s["route"] == route]
            if er:
                Xext = np.stack([base.scorer_features(s["data"]) for s in er])
                yext = np.asarray([s["label"] for s in er], dtype=object)
                known = np.isin(yext, np.unique(legacy_y))
                if known.any():
                    Xtr = np.concatenate([Xtr, Xext[known]], axis=0)
                    ytr = np.concatenate([ytr, yext[known]], axis=0)

            if len(Xtr) >= 20 and len(np.unique(ytr)) >= 2:
                route_mode = mode if route == "RL" else "baseline"
                models[route] = fit_linear_weighted(Xtr, ytr, route_mode, C=0.5)
        return models

    return fit_route_models


def evaluate_mode(mode, external, fold_assign, legacy_X, legacy_y, legacy_b):
    classes, families = calmod.build_vocab(legacy_y)
    original_fit = lc.fit_route_models
    lc.fit_route_models = make_fit_route_models(mode)

    try:
        pooled = Counter()
        routes_pooled = {r: Counter() for r in ("U", "N", "RL", "RH")}
        raw_rl = Counter()
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
            route_c = {r: Counter() for r in ("U", "N", "RL", "RH")}

            for s in test_rows:
                model = models.get(s["route"])
                if model is None or s["label"] not in model[1].classes_:
                    continue

                raw_rank, _ = calmod.raw_rank_scores(model, s["data"])
                if s["route"] == "RL":
                    raw_rl["n"] += 1
                    try:
                        pos = list(raw_rank).index(s["label"]) + 1
                    except ValueError:
                        pos = 999
                    raw_rl["top1"] += int(pos == 1)
                    raw_rl["top3"] += int(pos <= 3)
                    raw_rl["top5"] += int(pos <= 5)

                _, hybrid = tri.make_hybrid(model, cal, s, classes, families)
                gated = gate050.apply_gated(triad_cal, model, s, hybrid)
                sig = canon.apply_one(sig_cal, canon.SIG_PAIR, model, s, gated)
                out = canon.apply_guarded_gb(gb_cal, model, s, sig)

                hit = int(out and out[0] == s["label"])
                c["n"] += 1
                c["hits"] += hit
                route_c[s["route"]]["n"] += 1
                route_c[s["route"]]["hits"] += hit

            pooled.update(c)
            for route_name, values in route_c.items():
                routes_pooled[route_name].update(values)

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
                route_name: {
                    "n": values["n"],
                    "hits": values["hits"],
                    "top1": values["hits"] / max(1, values["n"]),
                }
                for route_name, values in routes_pooled.items()
                if values["n"]
            },
            "rl_raw_retrieval": {
                "n": raw_rl["n"],
                "top1": raw_rl["top1"] / max(1, raw_rl["n"]),
                "top3": raw_rl["top3"] / max(1, raw_rl["n"]),
                "top5": raw_rl["top5"] / max(1, raw_rl["n"]),
            },
            "folds": folds,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        lc.fit_route_models = original_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()

    state = joblib.load(args.state)
    external = reconstruct_external(state)
    legacy_rows = raw_legacy_rows()
    legacy_X, legacy_y, legacy_b = legacy_arrays(legacy_rows)

    paths = sorted(set(s["warc_path"] for s in external))
    fold_assign = {
        p: int.from_bytes(
            hashlib.sha256(("learning-curve:" + p).encode()).digest()[:8], "big"
        ) % OUTER_FOLDS
        for p in paths
    }

    results = {
        mode: evaluate_mode(mode, external, fold_assign, legacy_X, legacy_y, legacy_b)
        for mode in MODES
    }

    baseline = results["baseline"]
    deltas = {}
    for mode, candidate in results.items():
        brl = baseline["routes"].get("RL", {})
        crl = candidate["routes"].get("RL", {})
        deltas[mode] = {
            "hits": candidate["hits"] - baseline["hits"],
            "top1_pp": (candidate["top1"] - baseline["top1"]) * 100.0,
            "rl_hits": crl.get("hits", 0) - brl.get("hits", 0),
            "rl_top1_pp": (
                crl.get("top1", 0.0) - brl.get("top1", 0.0)
            ) * 100.0,
            "rl_raw_top3_pp": (
                candidate["rl_raw_retrieval"]["top3"]
                - baseline["rl_raw_retrieval"]["top3"]
            ) * 100.0,
        }

    print(json.dumps({
        "phase": "r10_paired_rl_class_balance",
        "results": results,
        "paired_deltas_vs_baseline": deltas,
        "selection_rule": (
            "Advance only if paired overall and RL final accuracy improve, "
            "with no loss of RL Top-3 retrieval."
        ),
        "methodology": {
            "locked_518_dim_features": True,
            "locked_hmt768": True,
            "only_rl_training_weights_change": True,
            "non_rl_fit_unchanged": True,
            "all_downstream_models_refit_under_each_arm": True,
            "same_reconstructed_rows": True,
            "same_fold_assignment": True,
            "cc_main_2026_30_used": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
