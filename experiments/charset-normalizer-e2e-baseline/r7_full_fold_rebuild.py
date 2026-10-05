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
MODE = "rl_hmt4096"


def hmt(data: bytes, total: int) -> bytes:
    if len(data) <= total:
        return data
    head = total // 3
    middle = total - 2 * head
    center = len(data) // 2
    left = center - middle // 2
    right = left + middle
    return data[:head] + data[left:right] + data[-head:]


def feature_from_sample(data: bytes, sampled: bytes) -> np.ndarray:
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


def candidate_features(data: bytes) -> np.ndarray:
    if base.route_bucket(data) == "RL":
        return feature_from_sample(data, hmt(data, 4096))
    return feature_from_sample(data, hmt(data, 768))


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
            if key in seen:
                continue
            seen.add(key)
            rows.append(s)
    return rows


def modified_legacy_training():
    rows, _ = collect_legacy()
    X = []
    y = []
    b = []
    for row in rows:
        data = row["path"].read_bytes()
        label = base.normalize_label(base.canonical_encoding(row["encoding"])) or row["encoding"]
        X.append(candidate_features(data))
        y.append(label)
        b.append(base.route_bucket(data))
    return np.stack(X), np.asarray(y, dtype=object), np.asarray(b, dtype=object)


def fingerprint(rows):
    h = hashlib.sha256()
    for s in sorted(rows, key=lambda x:(x["warc_path"], x.get("host",""), x["label"], len(x["data"]))):
        h.update(s["warc_path"].encode()); h.update(b"\0")
        h.update(s.get("host","").encode()); h.update(b"\0")
        h.update(s["label"].encode()); h.update(b"\0")
        h.update(hashlib.sha256(s["data"]).digest())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    args = parser.parse_args()

    state = joblib.load(args.state)
    external = reconstruct_external(state)
    if len(external) != 418:
        raise RuntimeError(f"expected canonical 418 reconstructed rows, got {len(external)}")

    original_features = base.scorer_features
    base.scorer_features = candidate_features
    try:
        t0 = time.perf_counter()
        legacy_X, legacy_y, legacy_b = modified_legacy_training()
        classes, families = calmod.build_vocab(legacy_y)

        paths = sorted(set(s["warc_path"] for s in external))
        fold_assign = {
            p: int.from_bytes(
                hashlib.sha256(("learning-curve:" + p).encode()).digest()[:8], "big"
            ) % OUTER_FOLDS
            for p in paths
        }

        pooled = Counter()
        folds = []

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
            train_seconds = time.perf_counter() - tf

            c = Counter()
            route_c = {"U": Counter(), "N": Counter(), "RL": Counter(), "RH": Counter()}
            for s in test_rows:
                model = models.get(s["route"])
                if model is None or s["label"] not in model[1].classes_:
                    continue
                _, hybrid = tri.make_hybrid(model, cal, s, classes, families)
                gated = gate050.apply_gated(triad_cal, model, s, hybrid)
                sig = canon.apply_one(sig_cal, canon.SIG_PAIR, model, s, gated)
                out = canon.apply_guarded_gb(gb_cal, model, s, sig)
                truth = s["label"]
                hit = int(out and out[0] == truth)
                c["n"] += 1
                c["hits"] += hit
                route_c[s["route"]]["n"] += 1
                route_c[s["route"]]["hits"] += hit

            folds.append({
                "fold": fold,
                "n": c["n"],
                "hits": c["hits"],
                "top1": c["hits"] / max(1, c["n"]),
                "training_seconds": train_seconds,
                "routes": {
                    r: {
                        "n": v["n"],
                        "hits": v["hits"],
                        "top1": v["hits"] / max(1, v["n"]),
                    }
                    for r, v in route_c.items()
                    if v["n"]
                },
            })
            pooled.update(c)
    finally:
        base.scorer_features = original_features

    result = {
        "phase": "r7_full_fold_rebuild",
        "candidate": MODE,
        "corpus_fingerprint": fingerprint(external),
        "canonical_expected_fingerprint": state["corpus_fingerprint"],
        "pooled": {
            "n": pooled["n"],
            "hits": pooled["hits"],
            "top1": pooled["hits"] / max(1, pooled["n"]),
            "baseline_hits": 365,
            "baseline_top1": 365 / 418,
            "delta_hits": pooled["hits"] - 365,
            "delta_top1_pp": ((pooled["hits"] / max(1, pooled["n"])) - (365 / 418)) * 100.0,
        },
        "folds": folds,
        "build_seconds": time.perf_counter() - t0,
        "methodology": {
            "rl_only_representation_change": True,
            "rl_sample": "HMT4096",
            "other_routes": "HMT768 unchanged",
            "all_route_models_retrained": True,
            "all_downstream_calibrators_retrained": True,
            "cc_main_2026_34_used_for_selection_only_not_this_fold_rebuild": True,
            "cc_main_2026_30_used": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if pooled["n"] != 418:
        raise SystemExit("canonical evaluated count changed")


if __name__ == "__main__":
    main()
