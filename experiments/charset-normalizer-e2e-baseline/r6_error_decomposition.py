from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import chardet
import joblib
from charset_normalizer import from_bytes

from codeccat import Detector
from codeccat.features import route
from codeccat.runtime import GB_PAIR, SIG_PAIR, _rule_rerank, build_context, family, replacement_rate

import release_candidate_independent_holdout as rc


def norm(label):
    return rc.normalize(label)


def top_label(rank):
    return norm(rank[0]) if rank else None


def stage_ranks(detector: Detector, data: bytes):
    rt = detector._runtime
    route_name = route(data)
    ctx = build_context(data, rt.bundle.route_models[route_name])

    raw = list(ctx.rank)
    rule = _rule_rerank(ctx)
    calibrated = rt._choose_cal(ctx)
    hybrid = rt._hybrid(ctx)
    gated = rt._gate(ctx, hybrid)
    sig = rt._pair(rt.bundle.utf8_sig, SIG_PAIR, ctx, gated)
    gb = rt._pair(rt.bundle.utf8_gb, GB_PAIR, ctx, sig)
    final = gb
    if (
        final and sig and final[0] != sig[0]
        and sig[0] == "gb18030" and final[0] == "utf-8"
        and replacement_rate(data) > 0.02
    ):
        final = sig

    return ctx, {
        "raw": raw,
        "rule": rule,
        "calibrated": calibrated,
        "hybrid": hybrid,
        "triad": gated,
        "utf8_sig": sig,
        "utf8_gb": gb,
        "final": final,
    }


def pct(n, d):
    return n / d if d else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--development-crawl", default="CC-MAIN-2026-34")
    args = parser.parse_args()

    state = joblib.load(args.state)
    canonical_dev = rc.unique_development_rows(state)
    bundle = rc.build_release_bundle(canonical_dev)
    detector = Detector(bundle)

    rows, _, collection_stats = rc.collect_holdout(args.development_crawl)

    stage_hits = Counter()
    transitions = {name: Counter() for name in (
        "rule", "calibrated", "hybrid", "triad", "utf8_sig", "utf8_gb", "final"
    )}
    raw_truth_rank = Counter()
    route_stats = defaultdict(Counter)
    family_stats = defaultdict(Counter)
    confusion = Counter()
    gap_confusion = Counter()
    cn_gap_by_family = Counter()
    cd_gap_by_family = Counter()
    cn_rescues = Counter()
    cd_rescues = Counter()
    final_error_examples = []

    previous_name = "raw"

    for row in rows:
        truth = norm(row["label"])
        ctx, stages = stage_ranks(detector, row["data"])
        n = 1
        for name, rank in stages.items():
            stage_hits[name] += int(top_label(rank) == truth)

        raw_norm = [norm(x) for x in stages["raw"]]
        try:
            raw_pos = raw_norm.index(truth) + 1
        except ValueError:
            raw_pos = 999
        if raw_pos == 1:
            raw_truth_rank["top1"] += 1
        elif raw_pos <= 3:
            raw_truth_rank["top3_not1"] += 1
        elif raw_pos <= 5:
            raw_truth_rank["top5_not3"] += 1
        elif raw_pos < 999:
            raw_truth_rank["below5"] += 1
        else:
            raw_truth_rank["absent"] += 1

        prev = "raw"
        for name in ("rule", "calibrated", "hybrid", "triad", "utf8_sig", "utf8_gb", "final"):
            prev_ok = top_label(stages[prev]) == truth
            now_ok = top_label(stages[name]) == truth
            if prev_ok and not now_ok:
                transitions[name]["harm"] += 1
            elif (not prev_ok) and now_ok:
                transitions[name]["help"] += 1
            elif top_label(stages[prev]) != top_label(stages[name]):
                transitions[name]["changed_neutral"] += 1
            prev = name

        final_pred = top_label(stages["final"])
        cn_match = from_bytes(row["data"]).best()
        cn_pred = norm(cn_match.encoding if cn_match is not None else None)
        cd_pred = norm(chardet.detect(row["data"]).get("encoding"))

        route_stats[ctx.route]["n"] += 1
        route_stats[ctx.route]["cc_hit"] += int(final_pred == truth)
        route_stats[ctx.route]["cn_hit"] += int(cn_pred == truth)
        route_stats[ctx.route]["cd_hit"] += int(cd_pred == truth)

        fam = family(truth)
        family_stats[fam]["n"] += 1
        family_stats[fam]["cc_hit"] += int(final_pred == truth)
        family_stats[fam]["cn_hit"] += int(cn_pred == truth)
        family_stats[fam]["cd_hit"] += int(cd_pred == truth)

        if final_pred != truth:
            confusion[(truth, final_pred)] += 1
            if cn_pred == truth:
                gap_confusion[(truth, final_pred)] += 1
                cn_gap_by_family[fam] += 1
                cn_rescues[(truth, final_pred)] += 1
            if cd_pred == truth:
                cd_gap_by_family[fam] += 1
                cd_rescues[(truth, final_pred)] += 1
            if len(final_error_examples) < 40:
                final_error_examples.append({
                    "truth": truth,
                    "cc": final_pred,
                    "cn": cn_pred,
                    "chardet": cd_pred,
                    "route": ctx.route,
                    "family": fam,
                    "raw_top5": raw_norm[:5],
                    "raw_truth_rank": None if raw_pos == 999 else raw_pos,
                    "stages": {k: top_label(v) for k, v in stages.items()},
                    "host": row.get("host", ""),
                    "bytes": len(row["data"]),
                })

    total = len(rows)
    summary = {
        "phase": "r6_cross_crawl_error_decomposition",
        "development_crawl": args.development_crawl,
        "n": total,
        "collection_stats": dict(collection_stats),
        "stage_accuracy": {
            name: {"hits": stage_hits[name], "top1": pct(stage_hits[name], total)}
            for name in ("raw", "rule", "calibrated", "hybrid", "triad", "utf8_sig", "utf8_gb", "final")
        },
        "stage_transition_effect": {
            name: dict(transitions[name])
            for name in transitions
        },
        "raw_truth_rank": dict(raw_truth_rank),
        "raw_truth_retrieval": {
            "top1": pct(raw_truth_rank["top1"], total),
            "top3": pct(raw_truth_rank["top1"] + raw_truth_rank["top3_not1"], total),
            "top5": pct(
                raw_truth_rank["top1"] + raw_truth_rank["top3_not1"] + raw_truth_rank["top5_not3"],
                total,
            ),
        },
        "by_route": {
            route: {
                **dict(stats),
                "cc_top1": pct(stats["cc_hit"], stats["n"]),
                "cn_top1": pct(stats["cn_hit"], stats["n"]),
                "chardet_top1": pct(stats["cd_hit"], stats["n"]),
            }
            for route, stats in sorted(route_stats.items())
        },
        "by_truth_family": {
            fam: {
                **dict(stats),
                "cc_top1": pct(stats["cc_hit"], stats["n"]),
                "cn_top1": pct(stats["cn_hit"], stats["n"]),
                "chardet_top1": pct(stats["cd_hit"], stats["n"]),
            }
            for fam, stats in sorted(family_stats.items())
        },
        "top_codeccat_confusions": [
            {"truth": t, "pred": p, "n": n}
            for (t, p), n in confusion.most_common(20)
        ],
        "top_charset_normalizer_gap_confusions": [
            {"truth": t, "cc_pred": p, "n": n}
            for (t, p), n in gap_confusion.most_common(20)
        ],
        "cn_gap_by_family": dict(cn_gap_by_family),
        "chardet_gap_by_family": dict(cd_gap_by_family),
        "charset_normalizer_rescues": [
            {"truth": t, "cc_pred": p, "n": n}
            for (t, p), n in cn_rescues.most_common(20)
        ],
        "chardet_rescues": [
            {"truth": t, "cc_pred": p, "n": n}
            for (t, p), n in cd_rescues.most_common(20)
        ],
        "error_examples": final_error_examples,
        "methodology": {
            "training": "canonical CC-MAIN-2026-39 development only",
            "analysis_data": "CC-MAIN-2026-34, already reclassified as development data",
            "untouched_release_holdout_used": False,
            "cc_main_2026_30_used": False,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
