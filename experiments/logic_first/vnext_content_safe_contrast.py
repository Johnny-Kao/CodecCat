"""One controlled accuracy experiment: document-disjoint folds plus contrastive reencoding.

No fixture-driven special-case rules, no L3, no performance optimization.
Candidate is development-only: evaluated original bytes must never contribute to
training by source group OR decoded-document identity. No model is promoted.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import codecs
from hashlib import sha256
import json
from pathlib import Path
import sys

from codeccat.vnext_strong_l2 import TrainableDetector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_training_strategy_audit import strict_text, examine, family

PAIRS = (
    ("cp037", "cp500"),
    ("iso8859-16", "iso8859-2"),
    ("iso8859-2", "cp1250"),
    ("cp1252", "iso8859-1"),
    ("cp1252", "iso8859-15"),
)
SEED = "codeccat-content-safe-v1"


def text_id(raw, label):
    text = strict_text(raw, label)
    if text is None:
        raise ValueError("invalid labeled prefix")
    return sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def component_folds(rows):
    """Union source-group & identical decoded-text neighbors before splitting."""
    parent = list(range(len(rows)))
    sizes = [1] * len(rows)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(a, b):
        a, b = root(a), root(b)
        if a == b:
            return
        if sizes[a] < sizes[b]:
            a, b = b, a
        parent[b] = a
        sizes[a] += sizes[b]
    groups, texts = {}, {}
    for i, (data, label, group, fold, digest) in enumerate(rows):
        d = text_id(data, label)
        if group in groups:
            join(i, groups[group])
        else:
            groups[group] = i
        if d in texts:
            join(i, texts[d])
        else:
            texts[d] = i
    comps = defaultdict(list)
    for i in range(len(rows)):
        comps[root(i)].append(i)
    ordered = sorted(comps.values(), key=lambda c: (
        -len(c), sha256((SEED + min(rows[i][4] for i in c)).encode()).hexdigest()))
    allocation = [0] * len(rows)
    loads = [0, 0, 0]
    for comp in ordered:
        fold = min(range(3), key=lambda x: (loads[x], x))
        for i in comp:
            allocation[i] = fold
        loads[fold] += len(comp)
    assert min(loads) > 0
    for k in range(3):
        train_idx = [i for i, f in enumerate(allocation) if f != k]
        held_idx = [i for i, f in enumerate(allocation) if f == k]
        assert {rows[i][2] for i in train_idx}.isdisjoint({rows[i][2] for i in held_idx})
        assert {text_id(rows[i][0], rows[i][1]) for i in train_idx}.isdisjoint(
            {text_id(rows[i][0], rows[i][1]) for i in held_idx})
        assert set(rows[i][4] for i in train_idx).isdisjoint(
            {rows[i][4] for i in held_idx})
    return allocation, {
        "component_count": len(comps),
        "n": len(rows),
        "fold_counts": loads,
        "max_component_size": max(len(c) for c in comps.values()),
        "source_group_isolation": True, "exact_text_isolation": True,
        "note": "near duplicates or genuinely related original documents can still cross splits",
    }


def contrastive_augmentation(training, per_direction=24, *, exclude_ebcdic=False):
    """Reencode training-only observed Unicode text into *different* codec bytes.

    Only trusted source labels and strictly encodeable target classes are used.
    Never create a conflicting label for identical bytes, nor fabricate text.
    Every generated example inherits the originating training group.
    """
    allowed = {y for raw, y, group in training}
    directed = set()
    for a, b in PAIRS:
        if exclude_ebcdic and {a, b} == {"cp037", "cp500"}:
            continue
        if a in allowed and b in allowed:
            directed.add((a, b))
            directed.add((b, a))
    existing = {sha256(raw).hexdigest(): label for raw, label, _ in training}
    selected = []
    counts = Counter()
    attempted = Counter()
    for source, target in sorted(directed):
        options = sorted((r for r in training if r[1] == source),
                         key=lambda r: sha256(r[0]).hexdigest())
        for raw, label, group in options:
            if counts[(source, target)] >= per_direction:
                break
            text = strict_text(raw, source)
            if text is None or len(text) < 12:
                continue
            try:
                encoded = text.encode(target, errors="strict")
                if encoded == raw or len(encoded) < 12:
                    continue
                test = strict_text(encoded, target)
                if test != text:
                    continue
            except (UnicodeError, LookupError, ValueError):
                continue
            attempted[(source, target)] += 1
            digest = sha256(encoded).hexdigest()
            if digest in existing:  # duplicated or conflicting label
                continue
            existing[digest] = target
            selected.append((encoded, target, group))
            counts[(source, target)] += 1
    return selected, {
        "n": len(selected),
        "generated_by_direction": {
            f"{a} -> {b}": counts[(a, b)] for a, b in sorted(directed)
        },
        "eligible_distinct_candidate_counts": {
            f"{a} -> {b}": attempted[(a, b)] for a, b in sorted(directed)
        },
        "generation": "strict decoded TRAIN text -> target codec -> strict round-trip; differing bytes",
    }


def batch_eval(models, held):
    records = []
    for raw, label, group, digest in held:
        truth = strict_text(raw, label)
        assert truth is not None
        records.append({
            "sha": digest, "truth": label, "group": group, "family": family(label),
            "predictions": {name: examine(model, raw, label, truth)
                            for name, model in models.items()},
        })
    return records


def summarize(cases):
    ret = {"n": len(cases)}
    for name in ("baseline", "contrastive", "latin_only"):
        items = [case["predictions"][name] for case in cases]
        ret[name] = {
            "correct_decoded_text": sum(x["text"] for x in items),
            "correct_label": sum(x["label"] for x in items),
            "abstention": sum(x["abstain"] for x in items),
            "truth_in_top3": sum(x["truth_in_top3"] for x in items),
        }
    ret["paired"] = {}
    for candidate in ("contrastive", "latin_only"):
        ret["paired"][candidate] = {}
        for metric in ("text", "label"):
            helped = sum(c["predictions"][candidate][metric] and
                         not c["predictions"]["baseline"][metric] for c in cases)
            harmed = sum(c["predictions"]["baseline"][metric] and
                         not c["predictions"][candidate][metric] for c in cases)
            ret["paired"][candidate][metric] = {"helped": helped, "harmed": harmed,
                                                "net": helped - harmed}
    ret["by_family"] = {}
    for fam in sorted(set(c["family"] for c in cases)):
        group = [c for c in cases if c["family"] == fam]
        ret["by_family"][fam] = {
            "n": len(group),
            "baseline_correct_text": sum(c["predictions"]["baseline"]["text"] for c in group),
            "contrastive_correct_text": sum(c["predictions"]["contrastive"]["text"] for c in group),
            "latin_only_correct_text": sum(c["predictions"]["latin_only"]["text"] for c in group),
        }
    ret["by_label"] = {}
    for label in sorted(set(c["truth"] for c in cases)):
        group = [c for c in cases if c["truth"] == label]
        ret["by_label"][label] = {
            "n": len(group),
            "baseline_correct_text": sum(c["predictions"]["baseline"]["text"] for c in group),
            "contrastive_correct_text": sum(c["predictions"]["contrastive"]["text"] for c in group),
        }
    return ret


def run(args):
    rows = corpus(Path(args.fixtures))
    allocation, split_report = component_folds(rows)
    all_cases = []
    folds = []
    for fold in range(3):
        train = [(raw, label, group) for (raw, label, group, _, digest), f
                 in zip(rows, allocation) if f != fold]
        held = [(raw, label, group, digest) for (raw, label, group, _, digest), f
                in zip(rows, allocation) if f == fold]
        baseline_train, dropped = residual_subset(train)
        augment, augment_report = contrastive_augmentation(
            baseline_train, per_direction=args.per_direction)
        latin_augment, latin_report = contrastive_augmentation(
            baseline_train, per_direction=args.per_direction, exclude_ebcdic=True)
        assert {group for _, _, group in augment + latin_augment}.issubset(
            {group for _, _, group in baseline_train})
        held_texts = {text_id(raw, label) for raw, label, _, _ in held}
        assert {text_id(raw, label) for raw, label, _ in augment + latin_augment}.isdisjoint(held_texts)
        models = {
            "baseline": TrainableDetector.fit(baseline_train, feature_profile="stable23"),
            "contrastive": TrainableDetector.fit(baseline_train + augment, feature_profile="stable23"),
            "latin_only": TrainableDetector.fit(baseline_train + latin_augment, feature_profile="stable23"),
        }
        cases = batch_eval(models, held)
        summary = summarize(cases)
        all_cases.extend(cases)
        folds.append({
            "fold": fold, "training_raw": len(train),
            "training_baseline": len(baseline_train),
            "training_candidate": len(baseline_train) + len(augment),
            "training_latin_only": len(baseline_train) + len(latin_augment),
            "residual_drop_by_label": dropped,
            "synthetic_training": augment_report,
            "latin_only_training": latin_report,
            "heldout": summary,
            "heldout_labels_not_trained": sorted(
                {y for _, y, _, _ in held} - {y for _, y, _ in baseline_train}),
        })
        print("CONTENT_SAFE_CONTRAST_FOLD", fold, json.dumps({
            "base_train": len(baseline_train),
            "new_train": len(baseline_train) + len(augment),
            "heldout": summary["n"], "paired": summary["paired"],
        }), flush=True)

    result = {
        "schema": "codeccat-content-disjoint-contrastive-audit-v1",
        "status": "development research only; NEVER direct release approval",
        "dataset": "chardet/test-data pinned commit",
        "model": "released stable23 runtime, identical feature profile, no L3",
        "folds": folds,
        "split": split_report,
        "aggregate": summarize(all_cases),
        "next_action": "If positive and not hiding regressions, use independent untouched external corpus before promotion. If negative, reject candidate without retuning against this same holdout.",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "paired.jsonl").write_text(
        "\n".join(json.dumps(case, sort_keys=True) for case in all_cases) + "\n")
    print("CONTENT_SAFE_CONTRAST_RESULT_BEGIN", flush=True)
    print(json.dumps({"split": split_report, "aggregate": result["aggregate"],
                      "folds": [{k: f[k] for k in ("fold","training_baseline","training_candidate","training_latin_only","synthetic_training","latin_only_training","heldout_labels_not_trained")}
                                for f in folds]}, indent=2, sort_keys=True), flush=True)
    print("CONTENT_SAFE_CONTRAST_RESULT_END", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--out", default="research-results/content-safe-contrast")
    parser.add_argument("--per-direction", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.per_direction <= 64:
        parser.error("per-direction must be 1..64")
    run(args)
