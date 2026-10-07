from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from codeccat import Detector, load_bundle
from codeccat.features import baseline_from_array, full_b_extra_from_array, hmt768_array
from codeccat.runtime import (
    InferenceContext,
    Runtime,
    _context,
    _raw,
    _signals,
)

import charset_external_scorer_domain_shift_ab as base

SAMPLE_BYTES = 4096
STRUCTURAL = ("big5", "gb18030", "shift-jis", "euc-jp", "euc-kr")
NEG = -1.0e30


def structural_mask(data: bytes) -> dict[str, bool]:
    # One bounded byte pass. Incomplete terminal multibyte state is treated as
    # unknown/possible because the 4096-byte observation window may cut a code unit.
    active = {
        "big5": True,
        "gb18030": True,
        "shift-jis": True,
        "euc-jp": True,
        "euc-kr": True,
    }
    b5 = gb = sj = ej = ek = 0

    for x in data[:SAMPLE_BYTES]:
        if active["big5"]:
            if b5 == 0:
                if x <= 0x7F:
                    pass
                elif 0x81 <= x <= 0xFE:
                    b5 = 1
                else:
                    active["big5"] = False
            else:
                if 0x40 <= x <= 0x7E or 0xA1 <= x <= 0xFE:
                    b5 = 0
                else:
                    active["big5"] = False

        if active["gb18030"]:
            if gb == 0:
                if x <= 0x7F:
                    pass
                elif 0x81 <= x <= 0xFE:
                    gb = 1
                else:
                    active["gb18030"] = False
            elif gb == 1:
                if 0x30 <= x <= 0x39:
                    gb = 2
                elif 0x40 <= x <= 0x7E or 0x80 <= x <= 0xFE:
                    gb = 0
                else:
                    active["gb18030"] = False
            elif gb == 2:
                if 0x81 <= x <= 0xFE:
                    gb = 3
                else:
                    active["gb18030"] = False
            else:
                if 0x30 <= x <= 0x39:
                    gb = 0
                else:
                    active["gb18030"] = False

        if active["shift-jis"]:
            if sj == 0:
                if x <= 0x7F or 0xA1 <= x <= 0xDF:
                    pass
                elif 0x81 <= x <= 0x9F or 0xE0 <= x <= 0xFC:
                    sj = 1
                else:
                    active["shift-jis"] = False
            else:
                if 0x40 <= x <= 0x7E or 0x80 <= x <= 0xFC:
                    sj = 0
                else:
                    active["shift-jis"] = False

        if active["euc-jp"]:
            if ej == 0:
                if x <= 0x7F:
                    pass
                elif x == 0x8E:
                    ej = 1
                elif x == 0x8F:
                    ej = 2
                elif 0xA1 <= x <= 0xFE:
                    ej = 4
                else:
                    active["euc-jp"] = False
            elif ej == 1:
                if 0xA1 <= x <= 0xDF:
                    ej = 0
                else:
                    active["euc-jp"] = False
            elif ej == 2:
                if 0xA1 <= x <= 0xFE:
                    ej = 3
                else:
                    active["euc-jp"] = False
            elif ej == 3:
                if 0xA1 <= x <= 0xFE:
                    ej = 0
                else:
                    active["euc-jp"] = False
            else:
                if 0xA1 <= x <= 0xFE:
                    ej = 0
                else:
                    active["euc-jp"] = False

        if active["euc-kr"]:
            if ek == 0:
                if x <= 0x7F:
                    pass
                elif 0xA1 <= x <= 0xFE:
                    ek = 1
                else:
                    active["euc-kr"] = False
            else:
                if 0xA1 <= x <= 0xFE:
                    ek = 0
                else:
                    active["euc-kr"] = False

        if not any(active.values()):
            break

    return active


def masked_context(data, signals, model, vector, mask):
    raw = _raw(model, vector)
    raw = np.asarray(raw, dtype=np.float64).copy()
    for i, cls in enumerate(model.classes):
        if cls in mask and not mask[cls]:
            raw[i] = NEG
    order = raw.argsort()[::-1]
    return InferenceContext(
        data=data,
        route=signals.route,
        rank=tuple(np.asarray(model.classes, dtype=object)[order]),
        sorted_scores=raw[order],
        raw_scores=raw,
        classes=model.classes,
        has_utf8_bom=signals.has_utf8_bom,
        strict_utf8=signals.strict_utf8,
        ascii_only=signals.ascii_only,
        high_byte_ratio=signals.high_byte_ratio,
        nul_ratio=signals.nul_ratio,
        sample_len_4k=signals.sample_len_4k,
    )


def rank_masked(rt: Runtime, data: bytes) -> tuple[str, ...]:
    signals = _signals(data)
    mask = structural_mask(data)
    array = hmt768_array(data)
    baseline_vector = baseline_from_array(array)

    bmodel = rt.bundle.baseline.route_models[signals.route]
    bctx = masked_context(data, signals, bmodel, baseline_vector, mask)
    baseline = rt.baseline.rank_context(bctx)

    threshold = rt.bundle.gate_thresholds.get(signals.route)
    margin = (
        float(bctx.sorted_scores[0] - bctx.sorted_scores[1])
        if len(bctx.sorted_scores) >= 2
        else float("inf")
    )

    if (
        rt.full_b is not None
        and threshold is not None
        and signals.route in ("U", "RH")
        and margin <= threshold
    ):
        extra = full_b_extra_from_array(array, signals.route)
        full_vector = np.concatenate([baseline_vector, extra])
        fmodel = rt.bundle.full_b.route_models[signals.route]
        fctx = masked_context(data, signals, fmodel, full_vector, mask)
        return tuple(rt.full_b.rank_context(fctx))

    return tuple(rt._apply_rl_specialist(bctx, baseline))


def normalize(label):
    return base.normalize_label(label)


def collect(crawl):
    old_crawl, old_url = base.CRAWL, base.WARC_PATHS_URL
    try:
        base.CRAWL = crawl
        base.WARC_PATHS_URL = f"https://data.commoncrawl.org/crawl-data/{crawl}/warc.paths.gz"
        rows, paths, stats = base.collect_external()
        return rows, paths, dict(stats)
    finally:
        base.CRAWL, base.WARC_PATHS_URL = old_crawl, old_url


def dedupe(rows):
    seen, out = set(), []
    for s in rows:
        key = (hashlib.sha256(s["data"]).digest(), normalize(s["label"]))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def evaluate(rows, detector, rt):
    base_hits = mask_hits = 0
    changed = beneficial = harmful = neutral = 0
    truth_masked = 0
    route = {r: Counter() for r in ("U", "N", "RL", "RH")}

    for s in rows:
        truth = normalize(s["label"])
        mask = structural_mask(s["data"])
        if truth in mask and not mask[truth]:
            truth_masked += 1

        b = detector.rank(s["data"])
        m = rank_masked(rt, s["data"])
        bh = int(bool(b) and b[0] == truth)
        mh = int(bool(m) and m[0] == truth)
        base_hits += bh
        mask_hits += mh
        route[s["route"]]["n"] += 1
        route[s["route"]]["base_hits"] += bh
        route[s["route"]]["mask_hits"] += mh

        if (b[0] if b else None) != (m[0] if m else None):
            changed += 1
            if mh and not bh:
                beneficial += 1
            elif bh and not mh:
                harmful += 1
            else:
                neutral += 1

    n = len(rows)
    return {
        "n": n,
        "baseline_hits": base_hits,
        "masked_hits": mask_hits,
        "baseline_top1": base_hits / max(1, n),
        "masked_top1": mask_hits / max(1, n),
        "delta_pp": 100.0 * (mask_hits - base_hits) / max(1, n),
        "changed_top1": changed,
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral": neutral,
        "truth_masked_n": truth_masked,
        "routes": {
            r: {
                "n": c["n"],
                "baseline_top1": c["base_hits"] / max(1, c["n"]),
                "masked_top1": c["mask_hits"] / max(1, c["n"]),
                "delta_hits": c["mask_hits"] - c["base_hits"],
            }
            for r, c in route.items() if c["n"]
        },
    }


def timing(rows, repeats=7):
    vals = []
    sink = 0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for s in rows:
            m = structural_mask(s["data"])
            sink += sum(m.values())
        vals.append((time.perf_counter_ns() - start) / max(1, len(rows)))
    return {
        "median_ns_per_sample": statistics.median(vals),
        "min_ns_per_sample": min(vals),
        "max_ns_per_sample": max(vals),
        "repeats": repeats,
        "sink": sink,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--expected-model-sha256", required=True)
    args = ap.parse_args()

    model = Path(args.model)
    actual = hashlib.sha256(model.read_bytes()).hexdigest()
    if actual != args.expected_model_sha256:
        raise RuntimeError(f"model SHA mismatch: {actual}")

    bundle = load_bundle(model)
    detector = Detector(bundle)
    rt = Runtime(bundle)

    crawls = ("CC-MAIN-2026-39", "CC-MAIN-2026-34")
    per_crawl, all_rows, collection = {}, [], {}
    for crawl in crawls:
        rows, paths, stats = collect(crawl)
        rows = dedupe(rows)
        all_rows.extend(rows)
        collection[crawl] = {"n": len(rows), "warc_paths_considered": len(paths), "stats": stats}
        per_crawl[crawl] = evaluate(rows, detector, rt)

    pooled = dedupe(all_rows)
    pooled_eval = evaluate(pooled, detector, rt)

    print(json.dumps({
        "phase": "h2_fused_candidate_mask_ab",
        "policy": {
            "r22_used": False,
            "parameters_tuned": False,
            "mask_source": "encoding-structure invariants only",
            "scan_bytes": SAMPLE_BYTES,
            "single_bounded_scan": True,
            "mask_position": "raw route-local scores before downstream policy",
            "release_candidate_modified": False,
        },
        "collection": collection,
        "per_crawl": per_crawl,
        "pooled": pooled_eval,
        "mask_only_timing": timing(pooled),
        "gate": {
            "safety_required": pooled_eval["truth_masked_n"] == 0,
            "no_harm_required": pooled_eval["harmful"] == 0,
            "cross_crawl_nonnegative_required": all(v["masked_hits"] >= v["baseline_hits"] for v in per_crawl.values()),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
