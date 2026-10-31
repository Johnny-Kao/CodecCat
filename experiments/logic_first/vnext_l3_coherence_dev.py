"""Source-group held-out comparison of optional decoded-language L3.

Uses only chardet/test-data development folds for fitting. Not a new blind
source and never an authorization to ship. Thresholds are fixed here before
this experiment's first CI evaluation; do not tune them on ICU confirmation.
"""
from __future__ import annotations

import argparse
import codecs
import collections
import json
import sys
from pathlib import Path

from codeccat.vnext_l3_language import LanguageCoherence
from codeccat.vnext_strong_l2 import TrainableDetector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset


def decoded(data: bytes, encoding: str | None) -> str | None:
    if encoding is None:
        return None
    try:
        d = codecs.getincrementaldecoder(encoding)("strict")
        return d.decode(data, final=len(data) < 4096)
    except (LookupError, UnicodeError, ValueError):
        return None


def run(args):
    rows = corpus(Path(args.fixtures))
    traces = []
    per_fold = []
    for fold in range(3):
        train = [(b, y, group) for b, y, group, k, _ in rows if k != fold]
        test = [(b, y, group, sha) for b, y, group, k, sha in rows if k == fold]
        assert {g for _, _, g in train}.isdisjoint({g for _, _, g, _ in test})
        trimmed, _ = residual_subset(train)
        l2 = TrainableDetector.fit(trimmed, feature_profile="stable23")
        l3 = LanguageCoherence.fit(train)
        counters = collections.Counter()
        for data, truth, group, sha in test:
            base = l2.rank(data, final=len(data) < 4096)
            proposed = l3.rerank(base, data, final=len(data) < 4096)
            target_text = decoded(data, truth)
            base_text = decoded(data, base.encoding)
            revised_text = decoded(data, proposed.encoding)
            base_ok = base_text is not None and base_text == target_text
            revised_ok = revised_text is not None and revised_text == target_text
            counters["n"] += 1
            counters["base_exact"] += base.encoding == truth
            counters["l3_exact"] += proposed.encoding == truth
            counters["base_text"] += bool(base_ok)
            counters["l3_text"] += bool(revised_ok)
            counters["l3_calls"] += proposed.called
            counters["l3_changes"] += proposed.changed
            counters["help"] += not base_ok and revised_ok
            counters["harm"] += base_ok and not revised_ok
            traces.append({
                "sha": sha, "fold": fold, "group": group, "truth": truth,
                "base": base.encoding, "l3": proposed.encoding,
                "base_text_ok": bool(base_ok), "l3_text_ok": bool(revised_ok),
                "called": proposed.called, "changed": proposed.changed,
                "reason": proposed.reason,
                "language_advantage": proposed.language_advantage,
            })
        per_fold.append({"fold": fold, "n_train": len(train), "n_l2": len(trimmed), **dict(counters)})
        print("L3_GROUPED_FOLD_DONE", fold, dict(counters), flush=True)
    summary = {name: sum(x.get(name, 0) for x in per_fold)
               for name in ("n", "base_exact", "l3_exact", "base_text", "l3_text",
                            "l3_calls", "l3_changes", "help", "harm")}
    report = {
        "schema": "vnext-l3-source-grouped-dev-v1",
        "source": "chardet/test-data, development and competitor-owned; NOT independent blind",
        "thresholds": {
            "max_l2_margin": 0.20,
            "min_language_advantage": 0.15,
            "minimum_trigrams": 12,
        },
        "l3_status": "OPTIONAL_DISABLED_BY_DEFAULT",
        "baseline_profile": "stable23 with L1 residual training",
        "summary": summary, "folds": per_fold,
        "interpretation": "Only a measured positive net decoded-text benefit on source-disjoint development AND subsequently untouched external sources could justify default-on. No promotion from this diagnostic alone.",
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "traces.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in traces) + "\n")
    print("L3_DEV_REPORT_BEGIN", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("L3_DEV_REPORT_END", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", required=True)
    p.add_argument("--output", default="research-results/vnext-l3-coherence")
    run(p.parse_args())
