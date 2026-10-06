from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import contextmanager

import joblib
import numpy as np

import charset_external_scorer_domain_shift_ab as base
import charset_candidate_calibration_ab as calmod
import r12_rl_candidate_specialist as r12
import r13_representation_sufficiency as r13
import r16a_fused_full_b as r16
import r17b_ambiguity_gated_full_b as r17

OUTER_FOLDS = 4
C_VALUES = (0.25, 0.50, 1.00)
R12_THRESHOLDS = (0.55, 0.65, 0.75)
GATE_FRACTIONS = (0.40, 0.50, 0.60)
CENTRAL_C = 0.50
CENTRAL_R12 = 0.65
CENTRAL_GATE = 0.50

EXPECTED_CENTRAL_BASELINE = 367
EXPECTED_CENTRAL_FULL = 378


@contextmanager
def primary_c(c):
    original = base.fit_linear

    def fixed_c(X, y, C=0.5):
        return original(X, y, C=c)

    base.fit_linear = fixed_c
    try:
        yield
    finally:
        base.fit_linear = original


def predict_with_threshold(stack, s, threshold):
    def _predict():
        model = stack["models"].get(s["route"])
        if model is None:
            return None, None
        rank, scores = calmod.raw_rank_scores(model, s["data"])
        margin = float(scores[0] - scores[1]) if len(scores) >= 2 else float("inf")
        _, hybrid = r17.tri.make_hybrid(
            model, stack["cal"], s, stack["classes"], stack["families"]
        )
        gated = r17.gate050.apply_gated(stack["triad"], model, s, hybrid)
        sig = r17.canon.apply_one(stack["sig"], r17.canon.SIG_PAIR, model, s, gated)
        out = r17.canon.apply_guarded_gb(stack["gb"], model, s, sig)
        if stack["spec"] is not None:
            out, _ = r12.apply_specialist(stack["spec"], model, s, out, threshold)
        return out, margin

    return r17.with_scorer(stack["scorer"], _predict)


def train_gate_thresholds(stack, train_pool):
    margins = defaultdict(list)

    def _collect():
        for s in train_pool:
            if s["route"] not in ("U", "RH"):
                continue
            model = stack["models"].get(s["route"])
            if model is None:
                continue
            _, scores = calmod.raw_rank_scores(model, s["data"])
            if len(scores) >= 2:
                margins[s["route"]].append(float(scores[0] - scores[1]))

    r17.with_scorer(stack["scorer"], _collect)
    thresholds = {}
    for frac in GATE_FRACTIONS:
        thresholds[frac] = {}
        for route in ("U", "RH"):
            vals = np.asarray(margins[route], dtype=np.float64)
            thresholds[frac][route] = (
                float(np.quantile(vals, frac)) if len(vals) else float("-inf")
            )
    return thresholds


def pack_hits(c):
    return {
        "n": c["n"],
        "hits": c["hits"],
        "top1": c["hits"] / max(1, c["n"]),
        "routes": {
            r: {
                "n": c[f"{r}_n"],
                "hits": c[f"{r}_hits"],
                "top1": c[f"{r}_hits"] / max(1, c[f"{r}_n"]),
            }
            for r in ("U", "N", "RL", "RH")
            if c[f"{r}_n"]
        },
    }


def add_hit(c, route, hit):
    c["n"] += 1
    c["hits"] += hit
    c[f"{route}_n"] += 1
    c[f"{route}_hits"] += hit


def suitability_plateau(values, center_key, tolerance_hits=1):
    ordered = [values[k]["hits"] for k in values]
    center = values[center_key]["hits"]
    return {
        "center_hits": center,
        "min_hits": min(ordered),
        "max_hits": max(ordered),
        "range_hits": max(ordered) - min(ordered),
        "center_within_one_hit_of_best": max(ordered) - center <= tolerance_hits,
        "plateau_within_two_hits_total": max(ordered) - min(ordered) <= 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()

    state = joblib.load(args.state)
    external = r17.reconstruct_external(state)
    legacy_rows = r17.raw_legacy_rows()

    base_X, legacy_y, legacy_b = r17.legacy_arrays(
        r13.baseline_features, legacy_rows
    )
    full_X, full_y, full_b = r17.legacy_arrays(
        r16.fused_full_b, legacy_rows
    )
    assert np.array_equal(legacy_y, full_y)
    assert np.array_equal(legacy_b, full_b)

    paths = sorted(set(s["warc_path"] for s in external))
    fold_assign = {
        p: int.from_bytes(
            hashlib.sha256(("learning-curve:" + p).encode()).digest()[:8], "big"
        ) % OUTER_FOLDS
        for p in paths
    }

    c_results = {
        str(c): {"baseline": Counter(), "full": Counter(), "folds": []}
        for c in C_VALUES
    }
    r12_results = {str(t): Counter() for t in R12_THRESHOLDS}
    gate_results = {str(f): Counter() for f in GATE_FRACTIONS}
    fold_rows = []

    for fold in range(OUTER_FOLDS):
        train_pool = [s for s in external if fold_assign[s["warc_path"]] != fold]
        test_rows = [s for s in external if fold_assign[s["warc_path"]] == fold]

        # Axis 1: primary scorer C. Everything else remains fixed.
        per_c_stacks = {}
        for c in C_VALUES:
            with primary_c(c):
                bstack = r17.fit_stack(
                    r13.baseline_features,
                    train_pool,
                    base_X,
                    legacy_y,
                    legacy_b,
                    True,
                )
                fstack = r17.fit_stack(
                    r16.fused_full_b,
                    train_pool,
                    full_X,
                    full_y,
                    full_b,
                    False,
                )
            per_c_stacks[c] = (bstack, fstack)

        # Central stacks are reused for R12 and gate suitability checks.
        central_b, central_f = per_c_stacks[CENTRAL_C]
        gate_thresholds = train_gate_thresholds(central_b, train_pool)

        fold_row = {
            "fold": fold,
            "c": {},
            "r12": {str(t): Counter() for t in R12_THRESHOLDS},
            "gate": {str(f): Counter() for f in GATE_FRACTIONS},
        }

        for s in test_rows:
            truth = s["label"]

            central_full_out = None
            central_base_by_t = {}
            central_margin = None

            for c in C_VALUES:
                bstack, fstack = per_c_stacks[c]
                bout, bmargin = predict_with_threshold(bstack, s, CENTRAL_R12)
                if bout is None:
                    continue
                bhit = int(bout[0] == truth)
                add_hit(c_results[str(c)]["baseline"], s["route"], bhit)

                fout = None
                if s["route"] in ("U", "RH"):
                    fout, _ = predict_with_threshold(fstack, s, CENTRAL_R12)
                chosen = fout if fout is not None else bout
                fhit = int(chosen and chosen[0] == truth)
                add_hit(c_results[str(c)]["full"], s["route"], fhit)

                if c == CENTRAL_C:
                    central_full_out = chosen
                    central_margin = bmargin

            # Axis 2: R12 threshold, using only the central baseline stack.
            for t in R12_THRESHOLDS:
                out, _ = predict_with_threshold(central_b, s, t)
                if out is None:
                    continue
                hit = int(out[0] == truth)
                add_hit(r12_results[str(t)], s["route"], hit)
                add_hit(fold_row["r12"][str(t)], s["route"], hit)
                central_base_by_t[t] = out

            # Axis 3: margin-gate fraction. Central C and R12 are locked.
            base_out = central_base_by_t.get(CENTRAL_R12)
            if base_out is not None:
                bhit = int(base_out[0] == truth)
                for frac in GATE_FRACTIONS:
                    c = gate_results[str(frac)]
                    fc = fold_row["gate"][str(frac)]
                    add_hit(c, s["route"], 0)
                    add_hit(fc, s["route"], 0)
                    # Undo hit placeholders; add_hit is reused for n/route accounting.
                    c["hits"] -= 0
                    c[f"{s['route']}_hits"] -= 0
                    fc["hits"] -= 0
                    fc[f"{s['route']}_hits"] -= 0

                    eligible = s["route"] in ("U", "RH")
                    if eligible:
                        c["eligible"] += 1
                        fc["eligible"] += 1
                    escalate = (
                        eligible
                        and central_margin is not None
                        and central_margin <= gate_thresholds[frac][s["route"]]
                    )
                    chosen = central_full_out if escalate else base_out
                    hit = int(chosen and chosen[0] == truth)
                    c["hits"] += hit
                    c[f"{s['route']}_hits"] += hit
                    fc["hits"] += hit
                    fc[f"{s['route']}_hits"] += hit
                    if escalate:
                        c["escalated"] += 1
                        fc["escalated"] += 1
                        if hit > bhit:
                            c["beneficial"] += 1
                            fc["beneficial"] += 1
                        elif hit < bhit:
                            c["harmful"] += 1
                            fc["harmful"] += 1

        # Fold summaries for C.
        for c in C_VALUES:
            b = c_results[str(c)]["baseline"]
            f = c_results[str(c)]["full"]
            # Per-fold values are reconstructed separately for audit clarity.
            with primary_c(c):
                bstack, fstack = per_c_stacks[c]
            fb = Counter()
            ff = Counter()
            for s in test_rows:
                bout, _ = predict_with_threshold(bstack, s, CENTRAL_R12)
                if bout is None:
                    continue
                add_hit(fb, s["route"], int(bout[0] == s["label"]))
                fout = None
                if s["route"] in ("U", "RH"):
                    fout, _ = predict_with_threshold(fstack, s, CENTRAL_R12)
                chosen = fout if fout is not None else bout
                add_hit(ff, s["route"], int(chosen and chosen[0] == s["label"]))
            fold_row["c"][str(c)] = {
                "baseline": pack_hits(fb),
                "full": pack_hits(ff),
            }

        fold_rows.append({
            "fold": fold,
            "c": fold_row["c"],
            "r12": {k: pack_hits(v) for k, v in fold_row["r12"].items()},
            "gate": {
                k: {
                    **pack_hits(v),
                    "eligible": v["eligible"],
                    "escalated": v["escalated"],
                    "beneficial": v["beneficial"],
                    "harmful": v["harmful"],
                }
                for k, v in fold_row["gate"].items()
            },
            "gate_thresholds": {
                str(frac): gate_thresholds[frac] for frac in GATE_FRACTIONS
            },
        })

    c_packed = {
        k: {
            "baseline": pack_hits(v["baseline"]),
            "full": pack_hits(v["full"]),
        }
        for k, v in c_results.items()
    }
    r12_packed = {k: pack_hits(v) for k, v in r12_results.items()}
    gate_packed = {
        k: {
            **pack_hits(v),
            "eligible": v["eligible"],
            "escalated": v["escalated"],
            "escalation_rate_all": v["escalated"] / max(1, v["n"]),
            "escalation_rate_eligible": v["escalated"] / max(1, v["eligible"]),
            "beneficial": v["beneficial"],
            "harmful": v["harmful"],
        }
        for k, v in gate_results.items()
    }

    c_baseline_values = {k: v["baseline"] for k, v in c_packed.items()}
    c_full_values = {k: v["full"] for k, v in c_packed.items()}

    central_reproduction = {
        "baseline_hits": c_packed[str(CENTRAL_C)]["baseline"]["hits"],
        "full_hits": c_packed[str(CENTRAL_C)]["full"]["hits"],
    }
    central_reproduction["matches_expected_current_harness"] = (
        central_reproduction["baseline_hits"] == EXPECTED_CENTRAL_BASELINE
        and central_reproduction["full_hits"] == EXPECTED_CENTRAL_FULL
    )

    print(json.dumps({
        "phase": "r19_parameter_suitability_audit",
        "purpose": "Validate that current parameters sit on stable neighborhoods; do not select a new value by best observed development accuracy.",
        "central_reproduction": central_reproduction,
        "axes": {
            "primary_scorer_C": {
                "values": C_VALUES,
                "central": CENTRAL_C,
                "results": c_packed,
                "baseline_plateau": suitability_plateau(
                    c_baseline_values, str(CENTRAL_C)
                ),
                "full_plateau": suitability_plateau(
                    c_full_values, str(CENTRAL_C)
                ),
            },
            "r12_threshold": {
                "values": R12_THRESHOLDS,
                "central": CENTRAL_R12,
                "results": r12_packed,
                "plateau": suitability_plateau(
                    r12_packed, str(CENTRAL_R12)
                ),
            },
            "margin_gate_fraction": {
                "values": GATE_FRACTIONS,
                "central": CENTRAL_GATE,
                "results": gate_packed,
                "plateau": suitability_plateau(
                    gate_packed, str(CENTRAL_GATE)
                ),
            },
        },
        "folds": fold_rows,
        "pre_registered_interpretation": {
            "suitable": "Central value is within one hit of the local best, the three-point range is at most two hits, and no neighboring point reveals a sharp collapse.",
            "not_parameter_tuning": "Do not replace the central value merely because one neighbor is best by one hit. The objective is local stability and reproducibility.",
            "gate_note": "For the gate, compute is also reported. Accuracy stability is primary for suitability; lower escalation is descriptive, not a license to retune after seeing held-out truth.",
        },
        "methodology": {
            "one_axis_at_a_time": True,
            "same_outer_folds": True,
            "same_training_corpus": True,
            "no_new_features": True,
            "no_new_model_family": True,
            "cc_main_2026_30_used": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
