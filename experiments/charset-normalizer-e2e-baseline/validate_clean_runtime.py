from __future__ import annotations

import argparse
import json

import joblib
import numpy as np

from codeccat import Detector, LinearModel, RuntimeBundle

import charset_p2_multihop_kernel_tournament as p2
import charset_p8_context_metadata_tournament as p8
import charset_p14_calibrator_hotpath_tournament as p14
import charset_p15_scalar_score_tournament as p15


def as_model(fused):
    if fused is None:
        return None
    classes, weight, bias = fused
    return LinearModel(
        classes=tuple(str(x) for x in classes),
        weight=np.asarray(weight, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
    )


def bundle_from_fold(fd, classes, families):
    return RuntimeBundle(
        route_models={
            route: as_model(p2.fuse_scaler_linear(model_tuple))
            for route, model_tuple in fd["models"].items()
        },
        calibrator=as_model(p2.fuse_estimator(fd["cal"])),
        triad=as_model(p2.fuse_estimator(fd["triad_cal"])),
        utf8_sig=as_model(p2.fuse_estimator(fd["sig_cal"])),
        utf8_gb=as_model(p2.fuse_estimator(fd["gb_cal"])),
        classes=tuple(str(x) for x in classes),
        families=tuple(str(x) for x in families),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    args = parser.parse_args()

    state = joblib.load(args.state)
    assert state["schema_version"] == 1
    assert sum(len(fd["rows"]) for fd in state["folds"]) == 418

    classes = state["classes"]
    families = state["families"]

    total = 0
    hits = 0
    rank_mismatches = []
    route_mismatches = []

    for fd in state["folds"]:
        detector = Detector(bundle_from_fold(fd, classes, families))
        fused = {
            route: p2.fuse_scaler_linear(model_tuple)
            for route, model_tuple in fd["models"].items()
        }
        meta = {
            route: p8.RouteMeta(model_tuple)
            for route, model_tuple in fd["models"].items()
        }
        reference = p14.CalDownstream(
            fd["cal"],
            fd["triad_cal"],
            fd["sig_cal"],
            fd["gb_cal"],
            classes,
            families,
            fd["models"],
            True,
            scalar_triad=True,
            inline_pairs=False,
            mode="combined",
        )

        for index, sample in enumerate(fd["rows"]):
            ctx = p15.build_ctx(
                sample,
                fused[sample["route"]],
                meta[sample["route"]],
                "reduceat",
                "dot",
            )
            expected = tuple(reference.run(ctx))
            actual = detector.rank(sample["data"])

            total += 1
            hits += int(actual and actual[0] == sample["label"])

            from codeccat.features import route as package_route
            computed_route = package_route(sample["data"])
            if computed_route != sample["route"]:
                route_mismatches.append(
                    {
                        "fold": fd["fold"],
                        "index": index,
                        "expected": sample["route"],
                        "actual": computed_route,
                    }
                )

            if actual != expected:
                rank_mismatches.append(
                    {
                        "fold": fd["fold"],
                        "index": index,
                        "truth": sample["label"],
                        "expected": list(expected[:5]),
                        "actual": list(actual[:5]),
                    }
                )

    report = {
        "phase": "clean_runtime_equivalence",
        "corpus_fingerprint": state["corpus_fingerprint"],
        "n": total,
        "hits": hits,
        "top1": hits / max(1, total),
        "target_hits": 365,
        "rank_mismatch_n": len(rank_mismatches),
        "route_mismatch_n": len(route_mismatches),
        "rank_mismatches": rank_mismatches[:20],
        "route_mismatches": route_mismatches[:20],
        "accepted": (
            total == 418
            and hits == 365
            and not rank_mismatches
            and not route_mismatches
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
