from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

from codeccat import Detector, LinearModel, RuntimeBundle, RuntimeStack, load_bundle, save_bundle

import charset_candidate_calibration_ab as calmod
import charset_canonical_guarded_final_validation as canon
import charset_external_scorer_learning_curve as lc
import charset_triad_specialist_ab as tri
import charset_p2_multihop_kernel_tournament as p2
import r12_rl_candidate_specialist as r12
import r13_representation_sufficiency as r13
import r16a_fused_full_b as r16
import r17b_ambiguity_gated_full_b as r17

GATE_FRACTION = 0.50
R12_THRESHOLD = 0.65
OOF_FOLDS = 4


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


def collect_crawl(crawl):
    import charset_external_scorer_domain_shift_ab as base
    old_crawl = base.CRAWL
    old_url = base.WARC_PATHS_URL
    try:
        base.CRAWL = crawl
        base.WARC_PATHS_URL = f"https://data.commoncrawl.org/crawl-data/{crawl}/warc.paths.gz"
        rows, _, stats = base.collect_external()
        return rows, dict(stats)
    finally:
        base.CRAWL = old_crawl
        base.WARC_PATHS_URL = old_url


def deduplicate(rows):
    seen = set()
    out = []
    for s in rows:
        key = (
            hashlib.sha256(s["data"]).digest(),
            str(s["label"]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def fingerprint(rows):
    h = hashlib.sha256()
    for s in sorted(
        rows,
        key=lambda x: (
            x["warc_path"],
            x.get("host", ""),
            x["label"],
            hashlib.sha256(x["data"]).hexdigest(),
        ),
    ):
        h.update(s["warc_path"].encode())
        h.update(b"\0")
        h.update(s.get("host", "").encode())
        h.update(b"\0")
        h.update(s["label"].encode())
        h.update(b"\0")
        h.update(hashlib.sha256(s["data"]).digest())
    return h.hexdigest()


def fit_stack_all(scorer, dev_rows, legacy_X, legacy_y, legacy_b, with_rl_specialist):
    def _fit():
        classes, families = calmod.build_vocab(legacy_y)
        models = lc.fit_route_models(dev_rows, legacy_X, legacy_y, legacy_b)
        cal = calmod.crossfit_calibration_rows(
            dev_rows, legacy_X, legacy_y, legacy_b, classes, families
        )
        triad_cal = tri.crossfit_triad_rows(
            dev_rows, legacy_X, legacy_y, legacy_b, classes, families
        )
        sig_cal = canon.fit_one(
            canon.SIG_PAIR, dev_rows, legacy_X, legacy_y, legacy_b, classes, families
        )
        gb_cal = canon.fit_one(
            canon.GB_PAIR, dev_rows, legacy_X, legacy_y, legacy_b, classes, families
        )
        spec = (
            r12.fit_specialist(dev_rows, legacy_X, legacy_y, legacy_b)
            if with_rl_specialist
            else None
        )
        return {
            "scorer": scorer,
            "models": models,
            "cal": cal,
            "triad": triad_cal,
            "sig": sig_cal,
            "gb": gb_cal,
            "spec": spec,
            "classes": classes,
            "families": families,
        }
    return r17.with_scorer(scorer, _fit)


def oof_gate_thresholds(dev_rows, legacy_X, legacy_y, legacy_b):
    paths = sorted(set(s["warc_path"] for s in dev_rows))
    fold_assign = {
        p: int.from_bytes(
            hashlib.sha256(("release-gate-oof:" + p).encode()).digest()[:8], "big"
        ) % OOF_FOLDS
        for p in paths
    }
    margins = defaultdict(list)
    fold_counts = []

    for fold in range(OOF_FOLDS):
        train = [s for s in dev_rows if fold_assign[s["warc_path"]] != fold]
        test = [s for s in dev_rows if fold_assign[s["warc_path"]] == fold]

        def _fit_models():
            return lc.fit_route_models(train, legacy_X, legacy_y, legacy_b)
        models = r17.with_scorer(r13.baseline_features, _fit_models)

        counts = Counter()
        def _score():
            for s in test:
                route = s["route"]
                if route not in ("U", "RH"):
                    continue
                model = models.get(route)
                if model is None:
                    continue
                _, scores = calmod.raw_rank_scores(model, s["data"])
                if len(scores) < 2:
                    continue
                margins[route].append(float(scores[0] - scores[1]))
                counts[route] += 1
        r17.with_scorer(r13.baseline_features, _score)
        fold_counts.append({"fold": fold, "U": counts["U"], "RH": counts["RH"]})

    thresholds = {}
    for route in ("U", "RH"):
        vals = np.asarray(margins[route], dtype=np.float64)
        if not len(vals):
            raise RuntimeError(f"no OOF gate margins for {route}")
        thresholds[route] = float(np.quantile(vals, GATE_FRACTION))
    return thresholds, fold_counts, {k: len(v) for k, v in margins.items()}


def build_bundle(dev_rows, legacy_rows):
    base_X, legacy_y, legacy_b = r17.legacy_arrays(r13.baseline_features, legacy_rows)
    full_X, full_y, full_b = r17.legacy_arrays(r16.fused_full_b, legacy_rows)
    if not np.array_equal(legacy_y, full_y) or not np.array_equal(legacy_b, full_b):
        raise RuntimeError("baseline/full-B legacy label mismatch")

    thresholds, fold_counts, margin_counts = oof_gate_thresholds(
        dev_rows, base_X, legacy_y, legacy_b
    )

    baseline = fit_stack_all(
        r13.baseline_features, dev_rows, base_X, legacy_y, legacy_b, True
    )
    full_b = fit_stack_all(
        r16.fused_full_b, dev_rows, full_X, full_y, full_b, False
    )

    bundle = RuntimeBundle(
        baseline=as_stack(baseline),
        full_b=as_stack(full_b),
        rl_specialist=as_model(p2.fuse_estimator(baseline["spec"])),
        gate_thresholds=thresholds,
        classes=tuple(str(x) for x in baseline["classes"]),
        families=tuple(str(x) for x in baseline["families"]),
        r12_threshold=R12_THRESHOLD,
    )
    return bundle, {
        "gate_thresholds": thresholds,
        "gate_oof_fold_counts": fold_counts,
        "gate_margin_counts": margin_counts,
        "r12_training_rows": int(baseline["spec"][2]) if baseline["spec"] else 0,
        "r12_training_counts": dict(baseline["spec"][3]) if baseline["spec"] else {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--additional-development-crawl", action="append", default=[])
    ap.add_argument("--model-out", required=True)
    args = ap.parse_args()

    state = joblib.load(args.state)
    canonical = r17.reconstruct_external(state)

    repo_root = Path(__file__).resolve().parents[2]
    old_cwd = Path.cwd()
    os.chdir(repo_root)
    try:
        legacy_rows = r17.raw_legacy_rows()
    finally:
        os.chdir(old_cwd)

    extra_rows = []
    extra_stats = {}
    for crawl in args.additional_development_crawl:
        rows, stats = collect_crawl(crawl)
        extra_rows.extend(rows)
        extra_stats[crawl] = {
            "collected": len(rows),
            "stats": stats,
        }

    dev_rows = deduplicate(canonical + extra_rows)
    dev_fp = fingerprint(dev_rows)

    bundle, fit_meta = build_bundle(dev_rows, legacy_rows)

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(out, bundle)
    sha256 = hashlib.sha256(out.read_bytes()).hexdigest()

    loaded = load_bundle(out)
    smoke = Detector(loaded)
    smoke_predictions = []
    for s in dev_rows[:16]:
        smoke_predictions.append(smoke.detect(s["data"]).encoding)

    report = {
        "phase": "r21_train_frozen_release_artifact",
        "architecture": "R20 frozen cached gated full-B cascade",
        "development": {
            "canonical_crawl": "CC-MAIN-2026-39",
            "canonical_rows": len(canonical),
            "additional_crawls": args.additional_development_crawl,
            "additional": extra_stats,
            "combined_unique_rows": len(dev_rows),
            "fingerprint": dev_fp,
            "route_counts": dict(Counter(s["route"] for s in dev_rows)),
            "label_counts": dict(Counter(s["label"] for s in dev_rows).most_common()),
        },
        "fit": fit_meta,
        "artifact": {
            "path": str(out),
            "bytes": out.stat().st_size,
            "sha256": sha256,
            "schema": 2,
            "roundtrip_loaded": True,
            "smoke_predictions_n": len(smoke_predictions),
        },
        "frozen_parameters": {
            "scorer_C": 0.5,
            "gate_fraction": GATE_FRACTION,
            "r12_threshold": R12_THRESHOLD,
            "s3": 0.02,
            "hmt": 768,
            "full_b_routes": ["U", "RH"],
        },
        "methodology": {
            "all_eligible_development_rows_used_for_final_route_models": True,
            "gate_thresholds_from_development_oof_margins": True,
            "fresh_holdout_used": False,
            "cc_main_2026_30_used": False,
            "no_parameter_selection_from_release_holdout": True,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
