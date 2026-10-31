"""Frozen, source-overlap-screened cross-repository evaluation (never train on holdout).

The Charamel fixture suite is a third-party benchmark, not natural-frequency
field sampling. Its files may share origins with chardet/test-data. Record
overlap removals and do NOT call this a clean blind test without provenance.
"""
from __future__ import annotations
import argparse, codecs, collections, hashlib, json, sys
from pathlib import Path

from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_l3_language import LanguageCoherence
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset


def canonical(s):
    try:
        return codecs.lookup(s).name
    except (LookupError, TypeError):
        return None


def text(raw, codec, final):
    if codec is None:
        return None
    try:
        return codecs.getincrementaldecoder(codec)("strict").decode(raw, final=final)
    except (UnicodeError, LookupError, ValueError):
        return None


def fingerprint(txt):
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def load_fixtures(root, supported, original):
    train_raw = {hashlib.sha256(raw).hexdigest() for raw, _, _ in original}
    train_text = set()
    for raw, label, _ in original:
        output = text(raw, label, final=len(raw) < 4096)
        if output is not None:
            train_text.add(fingerprint(output))
    counts = collections.Counter()
    seen_raw, seen_text = set(), set()
    records = []
    for path in sorted((root / "tests/fixtures").glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in (".txt", ".xml", ".html", ".csv", ".srt"):
            continue
        counts["file"] += 1
        label = canonical(path.parent.name)
        if label not in supported:
            counts["unsupported_encoding"] += 1
            continue
        full = path.read_bytes()
        if not full:
            counts["empty"] += 1
            continue
        raw = full[:4096]
        final = len(full) <= 4096
        expected = text(raw, label, final)
        if expected is None:
            counts["invalid_label_or_prefix"] += 1
            continue
        digest = hashlib.sha256(raw).hexdigest()
        decoded_hash = fingerprint(expected)
        if digest in train_raw or decoded_hash in train_text:
            counts["training_exact_byte_or_text_overlap"] += 1
            continue
        if digest in seen_raw or decoded_hash in seen_text:
            counts["within_source_duplicate_byte_or_text"] += 1
            continue
        seen_raw.add(digest)
        seen_text.add(decoded_hash)
        records.append((digest, raw, label, expected, final, path.parent.name, path.name))
    return records, dict(counts)


def run(args):
    import chardet
    from charset_normalizer import from_bytes

    training = [(b, y, g) for b, y, g, _, _ in corpus(Path(args.training))]
    focused, removed = residual_subset(training)
    l2 = TrainableDetector.fit(focused, feature_profile="stable23")
    l3 = LanguageCoherence.fit(training)
    supported = set(str(x) for x in l2.classifier.classes_)
    records, counts = load_fixtures(Path(args.holdout), supported, training)
    if len(records) < 25 or len({x[5] for x in records}) < 4:
        raise RuntimeError(f"too few non-overlapping supported fixtures: {len(records)} / {len(set(x[5] for x in records))} families, screening: {counts}")
    names = ("stable23", "stable23_l3", "chardet7", "charset_normalizer")
    metrics = {name: collections.Counter() for name in names}
    groups = collections.defaultdict(lambda: {name: collections.Counter() for name in names})
    traces = []
    for digest, raw, label, expected, final, family, filename in records:
        base = l2.rank(raw, final=final)
        reranked = l3.rerank(base, raw, final=final)
        cn = from_bytes(raw).best()
        predicted = {
            "stable23": base.encoding,
            "stable23_l3": reranked.encoding,
            "chardet7": canonical(chardet.detect(raw).get("encoding")),
            "charset_normalizer": canonical(cn.encoding if cn else None),
        }
        row = {"sha": digest, "label": label, "family": family, "file": filename,
               "bytes": len(raw), "final": final, "l3_called": reranked.called,
               "l3_changed": reranked.changed, "predictions": {}}
        for name in names:
            guess = canonical(predicted[name])
            result = text(raw, guess, final)
            ok = result is not None and result == expected
            exact = guess == label
            metrics[name]["n"] += 1
            metrics[name]["same_text"] += ok
            metrics[name]["exact"] += exact
            metrics[name]["abstain"] += result is None
            groups[family][name]["n"] += 1
            groups[family][name]["same_text"] += ok
            row["predictions"][name] = {"label": guess, "same_text": ok, "exact": exact}
        a = row["predictions"]["stable23"]["same_text"]
        b = row["predictions"]["stable23_l3"]["same_text"]
        metrics["stable23_l3"]["help"] += not a and b
        metrics["stable23_l3"]["harm"] += a and not b
        metrics["stable23_l3"]["called"] += reranked.called
        metrics["stable23_l3"]["changed"] += reranked.changed
        traces.append(row)
    report = {
        "schema": "vnext-charamel-overlap-screened-v1",
        "external_repo": "chomechome/charamel",
        "external_commit": args.holdout_commit,
        "license": "Apache-2.0 (repository license; individual upstream fixture provenance may vary)",
        "training_repo": "chardet/test-data",
        "training_commit": args.train_commit,
        "status": "EXTERNAL_DIAGNOSTIC_NOT_CONFIRMED_INDEPENDENT",
        "sample_warning": "Charamel fixtures may have common original sources with chardet; byte/text hashes cannot exclude near-duplicates. Capped 4096-byte prefixes and third-party labeled fixture folders are not production prevalence.",
        "screening": counts,
        "removed_residual_training": sum(removed.values()),
        "accepted_samples": len(records),
        "accepted_families": len({r[5] for r in records}),
        "metrics": {n: dict(v) for n,v in metrics.items()},
        "by_family": {family: {name: dict(k) for name,k in row.items()} for family,row in sorted(groups.items())},
        "decision_rule": "Do not enable L3 or promote any model from these results. A new independently sourced, preregistered confirmation is needed after development.",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+"\n")
    (output/"traces.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in traces)+"\n")
    print("EXTERNAL_CROSS_SOURCE_REPORT_BEGIN", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print("EXTERNAL_CROSS_SOURCE_REPORT_END", flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--training", required=True)
    p.add_argument("--holdout", required=True)
    p.add_argument("--train-commit", required=True)
    p.add_argument("--holdout-commit", required=True)
    p.add_argument("--output", default="research-results/vnext-charamel-cross-source")
    run(p.parse_args())
