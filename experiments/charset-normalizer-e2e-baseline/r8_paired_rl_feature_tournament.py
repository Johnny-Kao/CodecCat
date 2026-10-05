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
MODES = (
    "baseline",
    "rl_high16",
    "rl_highstats",
    "rl_high16_stats",
)


def hmt768(data: bytes) -> bytes:
    if len(data) <= 768:
        return data
    middle = len(data) // 2
    return data[:256] + data[middle - 128 : middle + 128] + data[-256:]


def base_features(data: bytes) -> np.ndarray:
    a = np.frombuffer(hmt768(data), dtype=np.uint8)
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


def high16_and_stats(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    a = np.frombuffer(data[:4096], dtype=np.uint8)
    high = a[a >= 128]
    high_n = len(high)

    if high_n:
        # 16 coarse bins over 0x80..0xFF, eight byte values per bin.
        idx = ((high.astype(np.uint16) - 128) >> 3).astype(np.intp)
        bins = np.bincount(idx, minlength=16).astype(np.float32)
        dist = bins / high_n
        nz = dist[dist > 0]
        entropy = float(-(nz * np.log2(nz)).sum() / 4.0)
        concentration = float(dist.max())
        c1_share = float(np.count_nonzero(high < 160) / high_n)
    else:
        dist = np.zeros(16, dtype=np.float32)
        entropy = 0.0
        concentration = 0.0
        c1_share = 0.0

    full_high_ratio = float(high_n / len(a)) if len(a) else 0.0
    stats = np.asarray(
        [full_high_ratio, c1_share, entropy, concentration],
        dtype=np.float32,
    )
    return dist, stats


def make_features(mode: str):
    extra_width = {
        "baseline": 0,
        "rl_high16": 16,
        "rl_highstats": 4,
        "rl_high16_stats": 20,
    }[mode]

    def scorer(data: bytes) -> np.ndarray:
        base_vec = base_features(data)
        if extra_width == 0:
            return base_vec

        # Keep non-RL arms behaviorally isolated: appended dimensions are zero.
        if base.route_bucket(data) != "RL":
            return np.concatenate(
                [base_vec, np.zeros(extra_width, dtype=np.float32)]
            )

        dist, stats = high16_and_stats(data)
        if mode == "rl_high16":
            extra = dist
        elif mode == "rl_highstats":
            extra = stats
        elif mode == "rl_high16_stats":
            extra = np.concatenate([dist, stats])
        else:
            raise ValueError(mode)
        return np.concatenate([base_vec, extra])

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
    scorer = make_features(mode)
    old_features = base.scorer_features
    base.scorer_features = scorer

    try:
        legacy_X, legacy_y, legacy_b = legacy_arrays(legacy_rows, scorer)
        classes, families = calmod.build_vocab(legacy_y)

        pooled = Counter()
        routes_pooled = {r: Counter() for r in ("U", "N", "RL", "RH")}
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
            "feature_width": len(scorer(b"example")),
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
            "folds": folds,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        base.scorer_features = old_features


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

    results = {}
    for mode in MODES:
        results[mode] = evaluate_mode(mode, external, fold_assign, legacy_rows)

    baseline = results["baseline"]
    deltas = {}
    for mode in MODES:
        candidate = results[mode]
        b_rl = baseline["routes"].get("RL", {})
        c_rl = candidate["routes"].get("RL", {})
        deltas[mode] = {
            "hits": candidate["hits"] - baseline["hits"],
            "top1_pp": (candidate["top1"] - baseline["top1"]) * 100.0,
            "rl_hits": c_rl.get("hits", 0) - b_rl.get("hits", 0),
            "rl_top1_pp": (
                c_rl.get("top1", 0.0) - b_rl.get("top1", 0.0)
            ) * 100.0,
        }

    print(json.dumps({
        "phase": "r8_paired_rl_feature_tournament",
        "reconstructed_corpus": {
            "n": len(external),
            "cached_state_external_n": state["external_n"],
            "note": (
                "All arms use the same 418 reconstructable rows, fold assignment, "
                "legacy rows, and fully refitted downstream pipeline."
            ),
        },
        "candidates": {
            "baseline": "locked HMT768 518-dim representation",
            "rl_high16": "RL adds 16-bin normalized distribution over 0x80..0xFF from first 4096 bytes",
            "rl_highstats": "RL adds full-4K high ratio, C1 share, normalized high-bin entropy, concentration",
            "rl_high16_stats": "RL adds both high16 and four summary stats",
        },
        "results": results,
        "paired_deltas_vs_baseline": deltas,
        "selection_rule": (
            "Advance only a candidate with positive paired overall and RL delta; "
            "prefer the smallest augmentation when gains are comparable."
        ),
        "methodology": {
            "same_reconstructed_rows": True,
            "same_fold_assignment": True,
            "same_legacy_rows": True,
            "all_models_and_downstream_refit_per_arm": True,
            "base_hmt768_preserved": True,
            "non_rl_extra_features_zeroed": True,
            "cc_main_2026_30_used": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
