from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import chardet
import numpy as np
from charset_normalizer import from_bytes

from codeccat import load_bundle
from codeccat.features import baseline_from_array, full_b_extra_from_array, hmt768_array
from codeccat.runtime import Runtime, _context, _signals, family
from h2_fused_candidate_mask_ab import collect, normalize

CRAWLS = ("CC-MAIN-2026-39", "CC-MAIN-2026-34", "CC-MAIN-2026-21",
          "CC-MAIN-2026-17", "CC-MAIN-2026-12", "CC-MAIN-2026-08")
BANNED = ("CC-MAIN-2026-30", "CC-MAIN-2026-25")


def trace(rt, data):
    signals = _signals(data)
    sampled = hmt768_array(data)
    vec = baseline_from_array(sampled)
    ctx = _context(data, signals, rt.bundle.baseline.route_models[signals.route], vec)
    stack = rt.baseline
    raw = ctx.rank
    rule = stack._hybrid(ctx)
    triad = stack._gate(ctx, rule)
    sig = stack._pair(stack.stack.utf8_sig, ("utf-8", "utf-8-sig"), ctx, triad)
    gb = stack._pair(stack.stack.utf8_gb, ("utf-8", "gb18030"), ctx, sig)
    baseline_final = stack.rank_context(ctx)
    margin = float(ctx.sorted_scores[0] - ctx.sorted_scores[1]) if len(ctx.sorted_scores) >= 2 else None
    threshold = rt.bundle.gate_thresholds.get(signals.route)
    escalation = (rt.full_b is not None and threshold is not None
                  and signals.route in ("U", "RH") and margin is not None and margin <= threshold)
    full_raw = full_final = None
    if rt.full_b is not None and signals.route in ("U", "RH"):
        full_vec = np.concatenate((vec, full_b_extra_from_array(sampled, signals.route)))
        full_ctx = _context(data, signals, rt.bundle.full_b.route_models[signals.route], full_vec)
        full_raw = full_ctx.rank[0]
        full_final = rt.full_b.rank_context(full_ctx)[0]
    final = rt.rank(data)[0]
    return {
        "route": signals.route, "high_byte_ratio": signals.high_byte_ratio,
        "nul_ratio": signals.nul_ratio, "strict_utf8": int(signals.strict_utf8),
        "utf8_bom": int(signals.has_utf8_bom), "sample_len": len(sampled),
        "raw_top1": raw[0], "raw_top2": raw[1] if len(raw) > 1 else "",
        "raw_top3": raw[2] if len(raw) > 2 else "",
        "truth_possible_top3": None, "margin": margin,
        "hybrid_top1": rule[0], "triad_top1": triad[0],
        "sig_top1": sig[0], "gb_top1": gb[0], "baseline_top1": baseline_final[0],
        "full_raw_top1": full_raw, "full_final_top1": full_final,
        "escalated": int(escalation), "final_top1": final,
    }


def smoke(rt):
    examples = (b"hello", b"A" * 512 + bytes([128]),
                bytes([128, 129]) * 300, bytes([0, 255]) * 300,
                bytes([239, 187, 191]) + b"hello",
                bytes([129, 64, 129]), bytes([255, 254, 0, 0]) + b"abc")
    got = set()
    for data in examples:
        tr = trace(rt, data)
        got.add(tr["route"])
        assert tr["final_top1"] == rt.rank(data)[0]
        assert tr["final_top1"] in rt.bundle.classes
        assert tr == trace(rt, data), "non-deterministic trace"
    assert set(("U", "N", "RL", "RH")).issubset(got), got
    print("H7_SMOKE_PASS", sorted(got), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--out", default="research-results/h7")
    a = p.parse_args()
    assert not set(CRAWLS).intersection(BANNED)
    model = Path(a.model)
    assert hashlib.sha256(model.read_bytes()).hexdigest() == a.sha256
    rt = Runtime(load_bundle(model))
    smoke(rt)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = {}
    conflicts = []
    collection = {}
    for crawl in CRAWLS:
        samples, paths, stats = collect(crawl)
        collection[crawl] = {"warc_paths": len(paths), "stats": stats, "accepted": len(samples)}
        for sample in samples:
            data = sample["data"]
            truth = normalize(sample["label"])
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                if seen[digest] != truth:
                    conflicts.append({"hash": digest, "first_truth": seen[digest], "second_truth": truth, "crawl": crawl})
                continue
            seen[digest] = truth
            tr = trace(rt, data)
            t0 = time.perf_counter_ns()
            pred = rt.rank(data)[0]
            elapsed = time.perf_counter_ns() - t0
            ch = chardet.detect(data).get("encoding")
            cn = from_bytes(data).best()
            tr.update({
                "crawl": crawl, "sha256": digest, "truth": truth,
                "family": family(truth), "byte_len": len(data),
                "tier": sample.get("tier"), "host_sha256": hashlib.sha256(sample.get("host", "").encode()).hexdigest()[:16],
                "warc_path": sample.get("warc_path", ""),
                "chardet7": normalize(ch) if ch else None,
                "charset_normalizer": normalize(cn.encoding) if cn else None,
                "r21_hit": int(pred == truth),
                "raw_hit": int(tr["raw_top1"] == truth),
                "raw_top3_hit": int(truth in (tr["raw_top1"], tr["raw_top2"], tr["raw_top3"])),
                "chardet_hit": int(normalize(ch) == truth) if ch else 0,
                "charset_hit": int(normalize(cn.encoding) == truth) if cn else 0,
                "inference_ns": elapsed,
            })
            rows.append(tr)
        print("H7_COLLECTED", crawl, len(samples), flush=True)

    assert rows
    with (out / "sample-trace.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def report(grouping):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[k] for k in grouping)].append(row)
        result = []
        for key, records in sorted(groups.items()):
            d = {"group": dict(zip(grouping, key)), "n": len(records)}
            for fld in ("r21_hit", "raw_hit", "raw_top3_hit", "chardet_hit", "charset_hit"):
                d[fld] = sum(x[fld] for x in records)
            d["r21_wrong_chardet_correct"] = sum(x["r21_hit"] == 0 and x["chardet_hit"] == 1 for x in records)
            d["r21_wrong_charset_correct"] = sum(x["r21_hit"] == 0 and x["charset_hit"] == 1 for x in records)
            d["raw_correct_final_wrong"] = sum(x["raw_hit"] == 1 and x["r21_hit"] == 0 for x in records)
            d["raw_wrong_final_correct"] = sum(x["raw_hit"] == 0 and x["r21_hit"] == 1 for x in records)
            result.append(d)
        return result

    metrics = {
        "phase": "h7_attribution_only", "crawls": CRAWLS,
        "selection_warning": "All six crawls are exploratory; selection is route capped and label curated.",
        "label_warning": "Collector does not retain raw header/body declaration; tier is available.",
        "data": collection, "n": len(rows), "exact_byte_label_conflicts": conflicts,
        "by_crawl": report(("crawl",)), "by_route": report(("route",)),
        "by_family": report(("family",)), "by_crawl_route_family": report(("crawl", "route", "family")),
        "by_route_family": report(("route", "family")), "by_tier": report(("tier",)),
        "runtime_ns": {"median": statistics.median(x["inference_ns"] for x in rows),
                       "p95": sorted(x["inference_ns"] for x in rows)[int(.95 * (len(rows)-1))]},
        "max_rss_kib": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss,
    }
    (out / "summary.json").write_text(json.dumps(metrics, indent=2))
    print("H7_RESULT_START")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("by_crawl_route_family",)}, indent=2))
    print("H7_RESULT_END")


if __name__ == "__main__":
    main()
