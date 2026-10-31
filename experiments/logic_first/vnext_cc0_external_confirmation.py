"""One-shot EXTERNAL CC0 generalization confirmation for frozen stable23 recipes.

This test reads pinned CC0 Tatoeba sentence exports, re-encodes ONE codec per
sentence, verifies independent overlap, and compares a fixed incumbent/candidate
against chardet 7.7.0 and charset-normalizer 3.5.2 on identical bytes.

The source archives & selected IDs were locked in run #41233758720 BEFORE
this script existed. No tuning is permitted after observing test outcomes.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import chardet
import charset_normalizer
from charset_normalizer import from_bytes
import numpy
import scipy
import sklearn

from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_decoding import decode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_content_safe_contrast import contrastive_augmentation
from vnext_cc0_freeze import SOURCES, SEED, MAX_PER_CODEC, get_archive, select, sha

EXPECTED = {
    "eng": "c2e2edd063a31cbbc8e416a4858225e934bbb0afff485031821e59f230ccf1ca",
    "fra": "cf4509f2f3fd542222be2cf085d0d814ce3f9405263c5746bcc965d7e83dfd08",
    "rus": "3b6645bbc18e4e3444ebc9195c4a6c41ff521a8bc4306fec8fcc217ae734db76",
}
EXPECTED_SELECTED = "6cbf9c040ea05bc295e3f6c4fd9f78bf92371bf6fbf63880d14c27de182f122e"
EXPECTED_TRAINING = "b0c0d206913ba118ffea2f7d306883e7a4a76db7"
EXPECTED_CH = "7.7.0"
EXPECTED_CN = "3.5.2"


def prepared_rows(training):
    train_texts = set()
    for raw, label, group in training:
        from vnext_training_strategy_audit import strict_text
        t = strict_text(raw, label)
        assert t is not None
        train_texts.add(sha(t.encode("utf-8", "surrogatepass")))
    all_raw, source_info = [], {}
    selected = []
    for lang in SOURCES:
        url, blob = get_archive(lang)
        digest = sha(blob)
        assert digest == EXPECTED[lang], f"{lang} Tatoeba archive changed; fail closed"
        rows, quota, eligible = select(blob, lang)
        selected.extend(rows)
        source_info[lang] = {"sha256":digest, "url":url, "n":len(rows),
                             "classes":quota,"eligible":eligible}
    assert sha(json.dumps(selected, ensure_ascii=False,sort_keys=True).encode("utf-8")) == EXPECTED_SELECTED
    seen_bytes, seen_text = set(), set()
    overlap = 0
    for row in selected:
        label = row["codec"]
        text = row["text"]
        raw = text.encode(label, "strict")
        assert raw.decode(label, "strict") == text
        assert sha(raw) == row["raw_sha256"]
        digest = row["text_sha256"]
        if digest in train_texts:
            overlap += 1
            continue
        if digest in seen_text:
            raise RuntimeError("External selection has repeated text")
        seen_text.add(digest)
        by = sha(raw)
        if by in seen_bytes:
            raise RuntimeError("External selection has duplicate bytes")
        seen_bytes.add(by)
        all_raw.append({"data":raw,"truth":label,"text":text,"lang":row["lang"],
                        "sentence_id":row["sentence_id"],"sha":by})
    assert len(all_raw) >= 450, ("too few independent external rows", len(all_raw))
    return all_raw, source_info, overlap


def strict_decode(raw, label):
    if not label:
        return None
    try:
        return raw.decode(label, "strict")
    except (UnicodeError, LookupError, ValueError, TypeError):
        return None


def detect_system(system, data):
    if system == "chardet7":
        return chardet.detect(data).get("encoding")
    if system == "charset_normalizer":
        result = from_bytes(data).best()
        return result.encoding if result is not None else None
    raise ValueError(system)


def evaluate(models, cases):
    results = []
    for case in cases:
        raw, truth, correct = case["data"], case["truth"], case["text"]
        outcomes = {}
        for name, model in models.items():
            result = decode(model, raw, bom_policy="strip")
            outcomes[name] = {
                "encoding": result.encoding,
                "text_correct": result.text == correct if result.text is not None else False,
                "label_correct": result.encoding == truth,
                "abstained": result.text is None,
                "ambiguous": result.ambiguous,
            }
        for name in ("chardet7", "charset_normalizer"):
            label = detect_system(name, raw)
            text = strict_decode(raw, label)
            outcomes[name] = {
                "encoding": label,
                "text_correct": text == correct if text is not None else False,
                "label_correct": label == truth,
                "abstained": text is None,
            }
        results.append({
            "sha":case["sha"],"lang":case["lang"],"sentence_id":case["sentence_id"],
            "truth":truth,"nbytes":len(raw),"outcomes":outcomes,
        })
    return results


def summary(rows):
    models = ("baseline","latin_only","chardet7","charset_normalizer")
    metrics = {}
    for name in models:
        metrics[name] = {
            "n":len(rows),
            "correct_decoded_text":sum(r["outcomes"][name]["text_correct"] for r in rows),
            "correct_label":sum(r["outcomes"][name]["label_correct"] for r in rows),
            "abstained":sum(r["outcomes"][name]["abstained"] for r in rows),
        }
    paired = {}
    for candidate in models[1:]:
        paired[candidate] = {}
        for m in ("text_correct","label_correct"):
            helped = sum(r["outcomes"][candidate][m] and not r["outcomes"]["baseline"][m] for r in rows)
            harmed = sum(r["outcomes"]["baseline"][m] and not r["outcomes"][candidate][m] for r in rows)
            paired[candidate][m] = {"helped":helped,"harmed":harmed,"net":helped-harmed}
    group_tables = {}
    for grouping in ("lang", "truth"):
        tables = {}
        for key in sorted(set(r[grouping] for r in rows)):
            subset = [r for r in rows if r[grouping] == key]
            tables[key] = {
                "n":len(subset),
                **{name:sum(r["outcomes"][name]["text_correct"] for r in subset) for name in models}
            }
        group_tables[grouping] = tables
    return {"metrics":metrics,"paired":paired,"group_tables":group_tables}


def run(args):
    import subprocess
    actual = subprocess.run(
        ["git","-C",args.fixtures,"rev-parse","HEAD"],check=True,
        capture_output=True,text=True
    ).stdout.strip()
    assert actual == EXPECTED_TRAINING, "unexpected training source SHA"
    assert chardet.__version__ == EXPECTED_CH
    assert charset_normalizer.__version__ == EXPECTED_CN
    training = [(b, label, group) for b, label, group, fold, digest in corpus(Path(args.fixtures))]
    cases, sources, contamination = prepared_rows(training)
    prior, removed = residual_subset(training)
    aug, evidence = contrastive_augmentation(prior, per_direction=24, exclude_ebcdic=True)
    models = {
        "baseline": TrainableDetector.fit(prior, feature_profile="stable23"),
        "latin_only": TrainableDetector.fit(prior + aug, feature_profile="stable23"),
    }
    results = evaluate(models, cases)
    answer = {
        "schema": "codeccat-cc0-frozen-one-shot-v1",
        "status": "CONSUMED_EXTERNAL_CONFIRMATION; NOT_AUTOMATIC_PROMOTION",
        "data": "Tatoeba author CC0 text reencoded synthetically; pinned SHA sources",
        "frozen_source_run": "https://github.com/Johnny-Kao/CodecCat/actions/runs/41233758720",
        "selection_digest": EXPECTED_SELECTED,
        "sources": sources,
        "n_before_training_overlap_filter": len(cases)+contamination,
        "training_content_overlaps_excluded": contamination,
        "training_origin": EXPECTED_TRAINING,
        "candidate": {
            "type":"strict-training-only-Latin-contrastive",
            "new_training_rows":len(aug), "training_rows":len(prior)+len(aug),
            "baseline_training_rows":len(prior), "augmentation":evidence,
        },
        "dependencies": {
            "python":platform.python_version(),"sklearn":sklearn.__version__,
            "scipy":scipy.__version__,"numpy":numpy.__version__,
            "chardet":chardet.__version__,"charset_normalizer":charset_normalizer.__version__
        },
        "quality":summary(results),
        "warning": [
            "External author text is independently sourced but encoding labels are synthetically generated",
            "This is one frozen suite of 3 correlated source languages; NOT naturally occurring files",
            "Several codec labels are unidentifiable from input bytes even when decoded text is correct",
            "No changes to Alpha, main or PyPI; an external real-byte holdout is still required for broad claims",
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(answer,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    (out/"paired.jsonl").write_text(
        "\n".join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in results)+"\n")
    print("CC0_ONE_SHOT_RESULT_BEGIN",flush=True)
    print(json.dumps({k:answer[k] for k in ("n_before_training_overlap_filter","training_content_overlaps_excluded","candidate","quality")},ensure_ascii=False,sort_keys=True,indent=2),flush=True)
    print("CC0_ONE_SHOT_RESULT_END",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fixtures",required=True)
    p.add_argument("--out",default="research-results/cc0-one-shot")
    run(p.parse_args())
