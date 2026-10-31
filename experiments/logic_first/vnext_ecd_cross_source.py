"""Independent repository smoke benchmark, not production-frequency blind evidence.

DereckLee/ecd fixture texts were synthetically encoded by that repository's
authors. The test is useful for cross-source behavior, but 1 text/codec and
its short handcrafted samples cannot prove broad real-world accuracy.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_l3_language import LanguageCoherence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_charamel_cross_source import canonical, fingerprint, text


def run(a):
    import chardet
    from charset_normalizer import from_bytes

    train = [(b, y, g) for b, y, g, _, _ in corpus(Path(a.training))]
    subset, _ = residual_subset(train)
    l2 = TrainableDetector.fit(subset, feature_profile="stable23")
    l3 = LanguageCoherence.fit(train)
    supported = set(str(c) for c in l2.classifier.classes_)
    known_bytes = {hashlib.sha256(b).hexdigest() for b, _, _ in train}
    known_text = {fingerprint(t) for b, y, _ in train
                  if (t := text(b, y, len(b) < 4096)) is not None}
    fixtures = json.loads((Path(a.fixtures)/"tests/fixtures/encodings.json").read_text())
    stats = collections.Counter()
    names = ("stable23", "stable23_l3", "chardet7", "charset_normalizer")
    metrics = {name:collections.Counter() for name in names}
    traces = []
    seen_bytes = set()
    for entry in fixtures:
        stats["listed"] += 1
        label = canonical(entry["expected"])
        if label not in supported:
            stats["untrained_label"] += 1
            continue
        raw_full = (Path(a.fixtures)/"tests/fixtures"/entry["file"]).read_bytes()
        if raw_full.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError("unexpected Git LFS pointer: "+entry["file"])
        raw = raw_full[:4096]
        final = len(raw_full) <= 4096
        expected = text(raw, label, final)
        if not raw or expected is None:
            stats["empty_or_invalid"] += 1
            continue
        digest=hashlib.sha256(raw).hexdigest()
        if digest in known_bytes or fingerprint(expected) in known_text:
            stats["overlap_train_byte_or_text"] += 1
            continue
        if digest in seen_bytes:
            stats["duplicate_within_fixtures"] += 1
            continue
        seen_bytes.add(digest)
        stats["accepted"] += 1
        if any(b >= 128 for b in raw):
            stats["contains_high_bit_bytes"] += 1
        base = l2.rank(raw, final=final)
        next_ = l3.rerank(base,raw,final=final)
        cn = from_bytes(raw).best()
        guesses = {
            "stable23":base.encoding,
            "stable23_l3":next_.encoding,
            "chardet7":canonical(chardet.detect(raw).get("encoding")),
            "charset_normalizer":canonical(cn.encoding if cn else None),
        }
        row={"file":entry["file"],"truth":label,"sha":digest,
             "l3_called":next_.called,"l3_changed":next_.changed,"predictions":{}}
        for name in names:
            pred = canonical(guesses[name])
            output = text(raw,pred,final)
            good=output is not None and output == expected
            metrics[name]["n"] += 1
            metrics[name]["same_text"] += good
            metrics[name]["exact"] += pred == label
            row["predictions"][name]={"encoding":pred,"same_text":good}
        original = row["predictions"]["stable23"]["same_text"]
        revised = row["predictions"]["stable23_l3"]["same_text"]
        metrics["stable23_l3"]["called"] += next_.called
        metrics["stable23_l3"]["changed"] += next_.changed
        metrics["stable23_l3"]["help"] += not original and revised
        metrics["stable23_l3"]["harm"] += original and not revised
        traces.append(row)
    if stats["accepted"] < 15 or stats["contains_high_bit_bytes"] < 10:
        raise RuntimeError(f"insufficient meaningful non-overlap: {dict(stats)}")
    report={
        "schema":"vnext-ecd-external-diagnostic-v1",
        "fixture_source":"DereckLee/ecd",
        "fixture_commit":a.commit,
        "license":"MIT",
        "description":"Independently authored, synthetically encoded per-codec short test strings; NOT natural field distribution",
        "training_source":"chardet/test-data",
        "training_commit":"b0c0d206913ba118ffea2f7d306883e7a4a76db7",
        "screening":dict(stats),
        "metrics":{name:dict(count) for name,count in metrics.items()},
        "caveat":"One example per supported codec, limited statistical power; should not be used to tune weights or call final product victory.",
    }
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    (dest/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (dest/"traces.jsonl").write_text("\n".join(json.dumps(r) for r in traces)+"\n")
    print("ECD_CROSS_SOURCE_REPORT_BEGIN",flush=True)
    print(json.dumps(report,indent=2,sort_keys=True),flush=True)
    print("ECD_CROSS_SOURCE_REPORT_END",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--training",required=True)
    p.add_argument("--fixtures",required=True)
    p.add_argument("--commit",required=True)
    p.add_argument("--output",default="research-results/vnext-ecd-external")
    run(p.parse_args())
