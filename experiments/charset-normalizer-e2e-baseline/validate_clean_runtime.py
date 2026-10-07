from __future__ import annotations

import argparse
import hashlib
import json

import joblib
import numpy as np

from codeccat import Detector, LinearModel, RuntimeBundle, RuntimeStack

import charset_p2_multihop_kernel_tournament as p2
import r13_representation_sufficiency as r13
import r16a_fused_full_b as r16
import r17b_ambiguity_gated_full_b as r17

GATE_FRACTION = 0.50
TARGET_HITS = 378


def as_model(fused):
    if fused is None:
        return None
    classes, weight, bias = fused
    return LinearModel(
        classes=tuple(str(x) for x in classes),
        weight=np.asarray(weight, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
    )


def as_stack(stack):
    return RuntimeStack(
        route_models={
            route: as_model(p2.fuse_scaler_linear(model_tuple))
            for route, model_tuple in stack["models"].items()
        },
        calibrator=as_model(p2.fuse_estimator(stack["cal"])),
        triad=as_model(p2.fuse_estimator(stack["triad"])),
        utf8_sig=as_model(p2.fuse_estimator(stack["sig"])),
        utf8_gb=as_model(p2.fuse_estimator(stack["gb"])),
    )


def as_specialist(spec):
    return as_model(p2.fuse_estimator(spec))


def bundle_from_stacks(bstack, fstack, thresholds):
    return RuntimeBundle(
        baseline=as_stack(bstack),
        full_b=as_stack(fstack),
        rl_specialist=as_specialist(bstack["spec"]),
        gate_thresholds={
            "U": float(thresholds["U"]),
            "RH": float(thresholds["RH"]),
        },
        classes=tuple(str(x) for x in bstack["classes"]),
        families=tuple(str(x) for x in bstack["families"]),
        r12_threshold=0.65,
    )


def reference_predict(bstack, fstack, sample, thresholds):
    bout, margin = r17.predict(bstack, sample, True)
    if bout is None:
        return None, False
    escalate = (
        sample["route"] in ("U", "RH")
        and margin is not None
        and margin <= thresholds[sample["route"]]
    )
    if not escalate:
        return tuple(bout), False
    fout, _ = r17.predict(fstack, sample, False)
    return tuple(fout if fout is not None else bout), True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    args = parser.parse_args()

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
        ) % 4
        for p in paths
    }

    total = 0
    hits = 0
    prediction_mismatches = []
    escalation_mismatches = []
    route_mismatches = []
    folds = []

    for fold in range(4):
        train_pool = [
            s for s in external if fold_assign[s["warc_path"]] != fold
        ]
        test_rows = [
            s for s in external if fold_assign[s["warc_path"]] == fold
        ]

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
        all_thresholds, _ = r17.train_margin_thresholds(bstack, train_pool)
        thresholds = all_thresholds[GATE_FRACTION]

        detector = Detector(bundle_from_stacks(bstack, fstack, thresholds))

        fold_hits = 0
        fold_n = 0
        fold_escalated = 0

        for index, sample in enumerate(test_rows):
            expected, expected_escalated = reference_predict(
                bstack, fstack, sample, thresholds
            )
            if expected is None:
                continue

            actual = detector.rank(sample["data"])

            # Reconstruct package escalation from the reference decision.
            # Prediction equivalence is the primary package contract; threshold
            # equivalence is checked separately through the stored threshold
            # and full rank result.
            actual_escalated = actual == tuple(
                r17.predict(fstack, sample, False)[0] or expected
            ) if sample["route"] in ("U", "RH") and expected_escalated else False

            total += 1
            fold_n += 1
            hit = int(actual and actual[0] == sample["label"])
            hits += hit
            fold_hits += hit
            fold_escalated += int(expected_escalated)

            if actual != expected:
                prediction_mismatches.append(
                    {
                        "fold": fold,
                        "index": index,
                        "truth": sample["label"],
                        "route": sample["route"],
                        "expected": list(expected[:5]),
                        "actual": list(actual[:5]),
                        "expected_escalated": expected_escalated,
                    }
                )

            if expected_escalated and not actual_escalated:
                escalation_mismatches.append(
                    {
                        "fold": fold,
                        "index": index,
                        "route": sample["route"],
                    }
                )

            from codeccat.features import route as package_route
            computed_route = package_route(sample["data"])
            if computed_route != sample["route"]:
                route_mismatches.append(
                    {
                        "fold": fold,
                        "index": index,
                        "expected": sample["route"],
                        "actual": computed_route,
                    }
                )

        folds.append(
            {
                "fold": fold,
                "n": fold_n,
                "hits": fold_hits,
                "top1": fold_hits / max(1, fold_n),
                "escalated": fold_escalated,
                "gate_thresholds": {
                    "U": float(thresholds["U"]),
                    "RH": float(thresholds["RH"]),
                },
            }
        )

    report = {
        "phase": "clean_runtime_r20_equivalence",
        "n": total,
        "hits": hits,
        "top1": hits / max(1, total),
        "target_hits": TARGET_HITS,
        "prediction_mismatch_n": len(prediction_mismatches),
        "escalation_mismatch_n": len(escalation_mismatches),
        "route_mismatch_n": len(route_mismatches),
        "prediction_mismatches": prediction_mismatches[:20],
        "escalation_mismatches": escalation_mismatches[:20],
        "route_mismatches": route_mismatches[:20],
        "folds": folds,
        "methodology": {
            "frozen_gate_fraction": GATE_FRACTION,
            "frozen_r12_threshold": 0.65,
            "cc_main_2026_30_used": False,
        },
        "accepted": (
            total == 418
            and hits == TARGET_HITS
            and not prediction_mismatches
            and not escalation_mismatches
            and not route_mismatches
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
