"""Source-group held-out training-strategy comparison for CodecCat stable23.

Development evidence ONLY: chardet/test-data is competitor-owned and is also
the Alpha model's training origin. No fresh external holdout or release claim.
This script does not change detector features, API, L3, or published weights.
"""
from __future__ import annotations

import argparse
import codecs
from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

from codeccat.vnext_bom import inspect_bom
from codeccat.vnext_strong_l2 import TrainableDetector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset

ARMS = ("full", "residual", "balanced")
SEED = 20261101


def balanced_subset(training, *, cap_per_label=48):
    """Deterministic, label-capped selection with source diversity first.

    No rows are synthesized/duplicated. A class below the cap remains intact.
    This is a trial sampling recipe, NOT an optimized hyperparameter.
    """
    by_label = defaultdict(lambda: defaultdict(list))
    for raw, label, group in training:
        by_label[label][group].append((raw, label, group))
    result = []
    for label in sorted(by_label):
        groups = by_label[label]
        queue = deque()
        for group in sorted(groups):
            ordered = sorted(groups[group], key=lambda row: sha256(row[0]).hexdigest())
            queue.append((group, deque(ordered)))
        chosen = 0
        while queue and chosen < cap_per_label:
            group, entries = queue.popleft()
            result.append(entries.popleft())
            chosen += 1
            if entries:
                queue.append((group, entries))
    assert {label for _, label, _ in result} == set(by_label)
    assert len(result) == len({sha256(raw).digest() for raw, _, _ in result})
    return result, {
        "cap_per_label": cap_per_label,
        "removed": len(training) - len(result),
        "before_by_label": dict(sorted(Counter(y for _, y, _ in training).items())),
        "after_by_label": dict(sorted(Counter(y for _, y, _ in result).items())),
        "source_groups": len(set(g for _, _, g in result)),
    }


def family(label):
    n = label.replace("_", "-") if label else ""
    if n.startswith("utf-"):
        return "unicode"
    if n in ("shift-jis", "cp932", "euc-jp", "euc-kr", "cp949",
             "big5", "gb18030", "gbk", "gb2312"):
        return "cjk"
    if n.startswith("iso8859-") or n.startswith("iso-8859-"):
        return "iso-8859"
    if n.startswith(("cp12", "windows-")):
        return "windows"
    return "other"


def strict_text(data, encoding):
    if encoding is None:
        return None
    try:
        dec = codecs.getincrementaldecoder(str(encoding))("strict")
        # corpus() may cap to 4096 bytes, where the last multibyte
        # character may be incomplete: compare strict decoded prefixes.
        text = dec.decode(data, final=len(data) < 4096)
        signature = inspect_bom(data, final=len(data) < 4096)
        if signature.signature is not None and text.startswith("\ufeff"):
            text = text[1:]
        return text
    except (UnicodeError, LookupError, ValueError):
        return None


def examine(model, raw, truth, truth_text):
    answer = model.rank(raw, final=len(raw) < 4096)
    prediction = str(answer.encoding) if answer.encoding is not None else None
    text = strict_text(raw, prediction)
    return {
        "pred": prediction,
        "label": prediction == truth,
        "text": text is not None and text == truth_text,
        "abstain": prediction is None or text is None,
        "truth_in_top3": truth in answer.candidates,
        "truth_supported": truth in model.classifier.classes_,
        "badly_decoded": prediction is not None and text is None,
    }


def rate(hits, count):
    return round(hits / count, 5) if count else None


def summarize(cases):
    n = len(cases)
    assert n
    overall = {}
    for arm in ARMS:
        results = [case["arms"][arm] for case in cases]
        overall[arm] = {
            "n": n,
            "correct_label": sum(x["label"] for x in results),
            "correct_text": sum(x["text"] for x in results),
            "label_accuracy": rate(sum(x["label"] for x in results), n),
            "text_accuracy": rate(sum(x["text"] for x in results), n),
            "abstain": sum(x["abstain"] for x in results),
            "truth_in_top3": sum(x["truth_in_top3"] for x in results),
            "truth_unseen_in_training": sum(not x["truth_supported"] for x in results),
            "bad_selected_decode": sum(x["badly_decoded"] for x in results),
        }
    pairs = {}
    for arm in ("residual", "balanced"):
        pairs[arm + "_vs_full"] = {
            metric: {
                "help": sum(c["arms"][arm][metric] and not c["arms"]["full"][metric] for c in cases),
                "harm": sum(c["arms"]["full"][metric] and not c["arms"][arm][metric] for c in cases),
            } for metric in ("label", "text")
        }
        for metric, record in pairs[arm + "_vs_full"].items():
            record["net"] = record["help"] - record["harm"]

    by_family = {}
    for fam in sorted({family(c["truth"]) for c in cases}):
        subset = [c for c in cases if family(c["truth"]) == fam]
        by_family[fam] = {
            "n": len(subset),
            "correct_text": {arm: sum(c["arms"][arm]["text"] for c in subset) for arm in ARMS},
            "correct_label": {arm: sum(c["arms"][arm]["label"] for c in subset) for arm in ARMS},
        }
    by_label = []
    for label, n_label in Counter(c["truth"] for c in cases).most_common():
        subset = [c for c in cases if c["truth"] == label]
        record = {"label": label, "n": n_label}
        for arm in ARMS:
            record[arm + "_wrong_text"] = sum(not c["arms"][arm]["text"] for c in subset)
        by_label.append(record)
    errors = {}
    for arm in ARMS:
        errors[arm] = [
            {"truth": y, "predicted": z, "n": count}
            for (y, z), count in Counter(
                (c["truth"], c["arms"][arm]["pred"])
                for c in cases if not c["arms"][arm]["text"]
            ).most_common(18)
        ]
    return {
        "overall": overall,
        "paired": pairs,
        "by_family": by_family,
        "worst_labels": sorted(by_label, key=lambda x: (-x["full_wrong_text"], x["label"]))[:25],
        "top_wrong_text_confusions": errors,
    }


def group_bootstrap(cases, arm, metric, samples=1000):
    """Group-resampled 95% CI for paired accuracy difference, not a p-value."""
    by_group = defaultdict(lambda: [0, 0])
    for case in cases:
        k = case["group"]
        by_group[k][0] += int(case["arms"][arm][metric]) - int(case["arms"]["full"][metric])
        by_group[k][1] += 1
    pairs = list(by_group.values())
    if len(pairs) < 2:
        return None
    rng = random.Random(SEED + (0 if metric == "text" else 1) + len(arm))
    values = []
    for _ in range(samples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        values.append(sum(x[0] for x in sample) / sum(x[1] for x in sample))
    values.sort()
    return {
        "lower": round(values[round((samples - 1) * .025)], 5),
        "upper": round(values[round((samples - 1) * .975)], 5),
        "groups": len(pairs),
        "resamples": samples,
    }


def run(args):
    rows = corpus(Path(args.fixtures))
    assert rows and len({r[4] for r in rows}) == len(rows)
    fingerprints = sha256()
    for raw, label, group, fold, digest in rows:
        fingerprints.update((digest + "|" + label + "|" + group + "|" + str(fold) + "\n").encode())
    all_cases = []
    fold_reports = []
    for fold in range(3):
        train = [(b, y, g) for b, y, g, k, _ in rows if k != fold]
        held = [(b, y, g, sha) for b, y, g, k, sha in rows if k == fold]
        assert {g for _, _, g in train}.isdisjoint({g for _, _, g, _ in held})
        residual, removed = residual_subset(train)
        balanced, balance_summary = balanced_subset(train, cap_per_label=args.label_cap)
        arms = {"full": train, "residual": residual, "balanced": balanced}
        labels = {y for _, y, _ in train}
        assert all({y for _, y, _ in sample} == labels for sample in arms.values())
        models = {}
        fit_probe = {}
        for name in ARMS:
            model = TrainableDetector.fit(arms[name], feature_profile="stable23")
            models[name] = model
            # In-sample probes use only examples actually in this arm.
            own_sample = sorted(arms[name], key=lambda r: sha256(r[0]).hexdigest())[:120]
            fit_probe[name] = {
                "n": len(own_sample),
                "text_correct": sum(
                    examine(model, raw, y, strict_text(raw, y))["text"]
                    for raw, y, _ in own_sample
                ),
            }
        cases = []
        for raw, truth, group, digest in held:
            correct_text = strict_text(raw, truth)
            assert correct_text is not None, ("invalid ground truth", digest, truth)
            cases.append({
                "sha": digest, "truth": truth, "group": group, "fold": fold,
                "size": len(raw),
                "arms": {name: examine(models[name], raw, truth, correct_text) for name in ARMS},
            })
        all_cases.extend(cases)
        fold_reports.append({
            "fold": fold, "training": {name: len(arms[name]) for name in ARMS},
            "training_sources": {name: len({g for _, _, g in arms[name]}) for name in ARMS},
            "labels_in_training": len(labels),
            "residual_removed_by_label": removed,
            "balancing": balance_summary,
            "train_fit_probe": fit_probe,
            "holdout": summarize(cases),
        })
        print("TRAINING_STRATEGY_FOLD_DONE", fold, json.dumps({
            "training": fold_reports[-1]["training"],
            "holdout": fold_reports[-1]["holdout"]["overall"],
        }), flush=True)

    total = summarize(all_cases)
    ci = {arm: {metric: group_bootstrap(all_cases, arm, metric)
                for metric in ("label", "text")}
          for arm in ("residual", "balanced")}
    report = {
        "schema": "codeccat-training-strategy-v1",
        "status": "development-diagnostic-not-blind-acceptance",
        "source": "chardet/test-data pinned revision; competitor-owned and training-origin",
        "dataset_fingerprint": fingerprints.hexdigest(),
        "n": len(all_cases),
        "groups": len(set(c["group"] for c in all_cases)),
        "split": "3 pre-existing hash(source_group) folds; same heldout rows across arms",
        "model": "actual TrainableDetector.fit(stable23), no feature/L3 change",
        "arms": {
            "full": "all available per-fold training rows",
            "residual": "existing L1-solved-row drop, <=35% per class, >=2 anchors",
            "balanced": "label cap with source-group round robin, cap not tuned to holdout",
        },
        "folds": fold_reports,
        "aggregate": total,
        "paired_group_bootstrap_95pct": ci,
        "limitations": [
            "Source stem groups are imperfect proxies for independent documents.",
            "Near-duplicate texts across distinct encoding/source groups are not removed.",
            "The dataset originates from a comparator and is not blind independent acceptance.",
            "The balancing cap is a diagnostic choice, not a tuned winner.",
            "Decoded-text equivalence uses the available <=4096 byte prefix and incremental strict decoding at boundary.",
            "In-sample probes are per-arm 120-row fit checks, NOT an unbiased estimate or a comparable train set.",
            "Repeated use of these held-out folds for selection would invalidate confirmation claims.",
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (out / "paired.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in all_cases) + "\n"
    )
    summary = {
        "n": report["n"], "groups": report["groups"], "aggregate": total,
        "group_bootstrap_ci": ci,
        "per_fold": [{ "fold": f["fold"], "n": f["holdout"]["overall"]["full"]["n"],
                       "training": f["training"], "holdout": f["holdout"]["overall"],
                       "train_fit_probe": f["train_fit_probe"] }
                     for f in fold_reports],
    }
    print("STRATEGY_AUDIT_JSON_START", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("STRATEGY_AUDIT_JSON_END", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--out", default="research-results/training-strategy-audit")
    parser.add_argument("--label-cap", type=int, default=48)
    args = parser.parse_args()
    if args.label_cap < 2:
        parser.error("label-cap must be >=2")
    run(args)
