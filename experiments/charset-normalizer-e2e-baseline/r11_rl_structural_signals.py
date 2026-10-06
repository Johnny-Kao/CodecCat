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
CANDIDATES = ("utf-8", "big5", "euc-kr", "cp1251", "iso-8859-1")


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


def decode_signals(data: bytes, encoding: str):
    sample = data[:4096]
    try:
        decoded = sample.decode(encoding, "strict")
        strict_valid = True
        replacement_rate = 0.0
    except UnicodeDecodeError:
        strict_valid = False
        decoded = sample.decode(encoding, "replace")
        replacement_rate = decoded.count("\ufffd") / max(1, len(decoded))

    # Round-trip mismatch catches permissive aliases and non-canonical replacement paths.
    try:
        roundtrip = decoded.encode(encoding, "replace")
        n = max(1, min(len(sample), len(roundtrip)))
        mismatch = sum(a != b for a, b in zip(sample[:n], roundtrip[:n])) / n
        mismatch += abs(len(sample) - len(roundtrip)) / max(1, len(sample))
    except Exception:
        mismatch = 1.0

    return {
        "strict_valid": strict_valid,
        "replacement_rate": replacement_rate,
        "roundtrip_mismatch": mismatch,
    }


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

    hard = []
    signal_summary = defaultdict(lambda: defaultdict(Counter))

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

        for s in test_rows:
            if s["route"] != "RL":
                continue
            model = models.get("RL")
            if model is None or s["label"] not in model[1].classes_:
                continue

            rank, scores = calmod.raw_rank_scores(model, s["data"])
            rank = list(rank)
            _, hybrid = tri.make_hybrid(model, cal, s, classes, families)
            gated = gate050.apply_gated(triad_cal, model, s, hybrid)
            sig = canon.apply_one(sig_cal, canon.SIG_PAIR, model, s, gated)
            final = canon.apply_guarded_gb(gb_cal, model, s, sig)
            pred = final[0] if final else None
            truth = s["label"]

            try:
                pos = rank.index(truth) + 1
            except ValueError:
                pos = 999

            # Aggregate signal separability for target classes.
            if truth in CANDIDATES:
                for enc in CANDIDATES:
                    sigs = decode_signals(s["data"], enc)
                    key = "truth_is_candidate" if enc == truth else "truth_is_other"
                    signal_summary[enc][key]["n"] += 1
                    signal_summary[enc][key]["strict_valid"] += int(sigs["strict_valid"])
                    signal_summary[enc][key]["replacement_rate_ppm_sum"] += int(round(sigs["replacement_rate"] * 1_000_000))
                    signal_summary[enc][key]["roundtrip_mismatch_ppm_sum"] += int(round(sigs["roundtrip_mismatch"] * 1_000_000))

            if pred != truth and pos > 5:
                signals = {enc: decode_signals(s["data"], enc) for enc in CANDIDATES}
                hard.append({
                    "fold": fold,
                    "truth": truth,
                    "final_pred": pred,
                    "raw_truth_rank": None if pos == 999 else pos,
                    "raw_top5": rank[:5],
                    "raw_top5_scores": [float(x) for x in scores[:5]],
                    "bytes": len(s["data"]),
                    "host": s.get("host", ""),
                    "signals": signals,
                })

    summarized = {}
    for enc, groups in signal_summary.items():
        summarized[enc] = {}
        for group, c in groups.items():
            n = c["n"]
            summarized[enc][group] = {
                "n": n,
                "strict_valid_rate": c["strict_valid"] / max(1, n),
                "mean_replacement_rate": c["replacement_rate_ppm_sum"] / max(1, n) / 1_000_000,
                "mean_roundtrip_mismatch": c["roundtrip_mismatch_ppm_sum"] / max(1, n) / 1_000_000,
            }

    print(json.dumps({
        "phase": "r11_rl_structural_signal_diagnostic",
        "hard_retrieval_failures": hard,
        "candidate_signal_summary": summarized,
        "interpretation_rule": (
            "Advance a specialist only if a small candidate-specific structural signal "
            "cleanly separates truth-class rows from competing RL rows; otherwise reject "
            "heuristic expansion."
        ),
        "methodology": {
            "locked_518_dim_features": True,
            "locked_hmt768": True,
            "full_downstream_refit": True,
            "same_reconstructed_rows": True,
            "same_fold_assignment": True,
            "signals_use_first_4096_bytes": True,
            "cc_main_2026_30_used": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
