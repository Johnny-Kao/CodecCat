from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict

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


def raw_rank(model, data):
    rank, scores = calmod.raw_rank_scores(model, data)
    return list(rank), [float(x) for x in scores]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()

    state = joblib.load(args.state)
    external = reconstruct_external(state)
    legacy_rows = raw_legacy_rows()

    legacy_X = np.stack([base.scorer_features(r["data"]) for r in legacy_rows])
    legacy_y = np.asarray([r["label"] for r in legacy_rows], dtype=object)
    legacy_b = np.asarray([r["route"] for r in legacy_rows], dtype=object)
    classes, families = calmod.build_vocab(legacy_y)

    paths = sorted(set(s["warc_path"] for s in external))
    fold_assign = {
        p: int.from_bytes(
            hashlib.sha256(("learning-curve:" + p).encode()).digest()[:8], "big"
        ) % OUTER_FOLDS
        for p in paths
    }

    pooled = Counter()
    truth_counts = Counter()
    pred_counts = Counter()
    confusion = Counter()
    raw_rank_bins = Counter()
    support = defaultdict(list)
    examples = []

    for fold in range(OUTER_FOLDS):
        train_pool = [s for s in external if fold_assign[s["warc_path"]] != fold]
        test_rows = [s for s in external if fold_assign[s["warc_path"]] == fold]
        ext_train = lc.deterministic_nested_subset(train_pool, TRAIN_N)

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

        rl_ext_train = Counter(s["label"] for s in ext_train if s["route"] == "RL")
        rl_legacy_train = Counter(
            label for label, route in zip(legacy_y, legacy_b) if route == "RL"
        )

        for label in set(rl_ext_train) | set(rl_legacy_train):
            support[label].append({
                "fold": fold,
                "external": rl_ext_train[label],
                "legacy": rl_legacy_train[label],
                "total": rl_ext_train[label] + rl_legacy_train[label],
            })

        for s in test_rows:
            if s["route"] != "RL":
                continue
            model = models.get("RL")
            if model is None or s["label"] not in model[1].classes_:
                continue

            rank, scores = raw_rank(model, s["data"])
            _, hybrid = tri.make_hybrid(model, cal, s, classes, families)
            gated = gate050.apply_gated(triad_cal, model, s, hybrid)
            sig = canon.apply_one(sig_cal, canon.SIG_PAIR, model, s, gated)
            final = canon.apply_guarded_gb(gb_cal, model, s, sig)

            truth = s["label"]
            pred = final[0] if final else None
            truth_counts[truth] += 1
            pred_counts[pred] += 1
            pooled["n"] += 1
            pooled["hits"] += int(pred == truth)
            if pred != truth:
                confusion[(truth, pred)] += 1

            try:
                pos = rank.index(truth) + 1
            except ValueError:
                pos = 999

            if pos == 1:
                raw_rank_bins["top1"] += 1
            elif pos <= 3:
                raw_rank_bins["top3_not1"] += 1
            elif pos <= 5:
                raw_rank_bins["top5_not3"] += 1
            elif pos < 999:
                raw_rank_bins["below5"] += 1
            else:
                raw_rank_bins["absent"] += 1

            if pred != truth:
                examples.append({
                    "fold": fold,
                    "truth": truth,
                    "final_pred": pred,
                    "raw_top5": rank[:5],
                    "raw_truth_rank": None if pos == 999 else pos,
                    "raw_margin_1_2": (scores[0] - scores[1]) if len(scores) >= 2 else None,
                    "bytes": len(s["data"]),
                    "host": s.get("host", ""),
                })

    n = pooled["n"]
    result = {
        "phase": "r9_rl_error_structure",
        "rl": {
            "n": n,
            "hits": pooled["hits"],
            "top1": pooled["hits"] / max(1, n),
            "raw_truth_retrieval": {
                "top1": raw_rank_bins["top1"] / max(1, n),
                "top3": (raw_rank_bins["top1"] + raw_rank_bins["top3_not1"]) / max(1, n),
                "top5": (
                    raw_rank_bins["top1"]
                    + raw_rank_bins["top3_not1"]
                    + raw_rank_bins["top5_not3"]
                ) / max(1, n),
                "bins": dict(raw_rank_bins),
            },
            "truth_counts": truth_counts.most_common(),
            "prediction_counts": pred_counts.most_common(),
            "top_confusions": [
                {"truth": t, "pred": p, "n": count}
                for (t, p), count in confusion.most_common(20)
            ],
            "errors": examples,
        },
        "training_support_by_truth": dict(sorted(support.items())),
        "methodology": {
            "same_reconstructed_418_rows": True,
            "same_fold_assignment_as_r7_r8": True,
            "locked_518_dim_features": True,
            "full_downstream_refit": True,
            "cc_main_2026_30_used": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
