"""One-time pre-registered ICU encoding-suite confirmation.

Never train, tune, select weights, or calibrate on this fixture.
See research/VNEXT_ICU_CONFIRMATION_PROTOCOL.md.
"""
from __future__ import annotations
import argparse
import codecs
import collections
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_multidist import family
from codeccat.vnext_strong_l2 import TrainableDetector


def canonical(name):
    try:
        return codecs.lookup(name).name
    except (LookupError, TypeError):
        return None


def text_from_bytes(data, encoding, final):
    if encoding is None:
        return None
    try:
        return codecs.getincrementaldecoder(encoding)("strict").decode(data, final=final)
    except (LookupError, UnicodeError, TypeError):
        return None


def run(args):
    import chardet
    from charset_normalizer import from_bytes

    original = [(b, y, g) for b, y, g, fold, sha in corpus(Path(args.train))]
    train, removed = residual_subset(original)
    models = {
        "stable23": TrainableDetector.fit(train, feature_profile="stable23"),
        "experimental13": TrainableDetector.fit(train, feature_profile="experimental13"),
    }
    supported = set(models["stable23"].classifier.classes_) & set(models["experimental13"].classifier.classes_)
    known_bytes = {hashlib.sha256(b).hexdigest() for b, _, _ in original}
    known_text = set()
    for b, y, _ in original:
        try:
            known_text.add(hashlib.sha256(b.decode(y).encode("utf-8")).hexdigest())
        except (UnicodeError, LookupError):
            pass

    tree = ET.parse(Path(args.fixture))
    cases = tree.findall(".//test-case")
    counts = collections.Counter()
    records = []
    seen = set()
    for case in cases:
        case_id = case.attrib.get("id", "unnamed")
        text = "".join(case.itertext()).strip()
        if not text:
            counts["empty_cases"] += 1
            continue
        for labeled in case.attrib.get("encodings", "").split():
            codec = canonical(labeled.split("/", 1)[0])
            counts["pairs_declared"] += 1
            if codec is None or codec not in supported:
                counts["unsupported_class"] += 1
                continue
            try:
                full = text.encode(codec, "strict")
            except (UnicodeError, LookupError):
                counts["not_encodable"] += 1
                continue
            raw = full[:4096]
            final = len(full) <= 4096
            expected = text_from_bytes(raw, codec, final)
            if not raw or expected is None:
                counts["invalid_or_empty"] += 1
                continue
            rawhash = hashlib.sha256(raw).hexdigest()
            texthash = hashlib.sha256(expected.encode("utf-8")).hexdigest()
            if rawhash in known_bytes or texthash in known_text:
                counts["training_overlap"] += 1
                continue
            key = (rawhash, codec)
            if key in seen:
                counts["within_source_duplicate"] += 1
                continue
            seen.add(key)
            records.append({
                "case": case_id,
                "label": codec,
                "family": family(codec),
                "raw": raw,
                "expected": expected,
                "final": final,
                "sha": rawhash,
            })
    if len(records) < 30 or len(set(r["case"] for r in records)) < 8:
        raise RuntimeError(f"insufficient independent ICU test coverage: {len(records)} rows")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    traces = []
    result_names = ("stable23", "experimental13", "chardet_7", "charset_normalizer")
    for r in records:
        raw = r["raw"]
        predictions = {
            name: canonical(model.rank(raw, final=r["final"]).encoding)
            for name, model in models.items()
        }
        predictions["chardet_7"] = canonical(chardet.detect(raw).get("encoding"))
        cn = from_bytes(raw).best()
        predictions["charset_normalizer"] = canonical(cn.encoding if cn else None)
        trace = {
            key: r[key] for key in ("case", "label", "family", "final", "sha")
        }
        trace["bytes"] = len(raw)
        trace["results"] = {}
        for name in result_names:
            predicted = predictions[name]
            decoded = text_from_bytes(raw, predicted, r["final"])
            trace["results"][name] = {
                "prediction": predicted,
                "exact": predicted == r["label"],
                "same_text": decoded is not None and decoded == r["expected"],
                "abstain": predicted is None or decoded is None,
            }
        traces.append(trace)

    group_ids = sorted({t["case"] for t in traces})
    metrics = {}
    for name in result_names:
        exact = sum(t["results"][name]["exact"] for t in traces)
        text_hits = sum(t["results"][name]["same_text"] for t in traces)
        abstained = sum(t["results"][name]["abstain"] for t in traces)
        case_scores = [
            sum(t["results"][name]["same_text"] for t in traces if t["case"] == cid)
            / sum(t["case"] == cid for t in traces)
            for cid in group_ids
        ]
        metrics[name] = {
            "exact": exact, "same_text": text_hits, "abstentions": abstained,
            "case_macro_text_rate": round(sum(case_scores) / len(case_scores), 5),
            "by_family": {
                fam: {
                    "n": sum(t["family"] == fam for t in traces),
                    "same_text": sum(t["results"][name]["same_text"]
                                     for t in traces if t["family"] == fam),
                }
                for fam in sorted({t["family"] for t in traces})
            },
        }
    report = {
        "fixture_repo": "unicode-org/icu",
        "fixture_commit": args.icu_commit,
        "source_path": "icu4c/source/test/testdata/csdetest.xml",
        "training_commit": args.train_commit,
        "protocol": "research/VNEXT_ICU_CONFIRMATION_PROTOCOL.md",
        "pre_registration": "before first run on this source",
        "source_test_cases": len(cases),
        "accepted_cases": len(group_ids),
        "accepted_samples": len(traces),
        "counts": dict(counts),
        "metrics": metrics,
        "caveat": "One-time ICU-authored test suite, correlated encodings per case. NOT natural field prevalence. Must not be used for further tuning or claim fully independent source replications.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "traces.jsonl").write_text("\n".join(map(json.dumps, traces)) + "\n")
    print("ICU_CONFIRM_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("ICU_CONFIRM_END")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--fixture", required=True)
    p.add_argument("--icu-commit", required=True)
    p.add_argument("--train-commit", required=True)
    p.add_argument("--output", default="research-results/vnext-icu-confirmation")
    run(p.parse_args())
