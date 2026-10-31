"""Accuracy experiment: *new CC0 training* with only identifiable codec evidence.

All Tatoeba examples are author CC0 and external to chardet/test-data. We
do not train on the previously frozen 560 CC0 confirmation sentences.
Performance claims and deployment decisions explicitly out of scope.
"""
from __future__ import annotations

import argparse
import bz2
import codecs
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

from codeccat.vnext_strong_l2 import TrainableDetector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_training_strategy_audit import strict_text, examine, family
from vnext_content_safe_contrast import component_folds, text_id
from vnext_cc0_freeze import SOURCES, sha, get_archive, select
from vnext_cc0_external_confirmation import EXPECTED, EXPECTED_SELECTED

TRAIN_SEED = "codeccat-new-cc0-discriminative-train-20261101-v1"
CAP = 120


def read_source_pool(lang, blob):
    lines = bz2.decompress(blob).decode("utf-8", "strict").splitlines()
    for line in lines:
        fields = line.split("\t")
        if len(fields) >= 3 and fields[1] == lang:
            sid, text = fields[0], fields[2]
        elif len(fields) >= 2:
            sid, text = fields[0], fields[1]
        else:
            continue
        if sid.isdecimal() and 48 <= len(text) <= 350 and "\x00" not in text and "\ufffd" not in text:
            yield sid, text


def trainable_evidence(lang, allowed, source_blob, excluded_ids, exclude_texts):
    """One target label per sentence, only if other declared targets differ in decoded text.

    A category label is not inferred from text when several codecs have
    exactly the same byte-to-Unicode interpretation: no random supervision
    of intrinsically ambiguous labels.
    """
    options = SOURCES[lang]
    records = sorted(read_source_pool(lang, source_blob),
                     key=lambda r: sha((TRAIN_SEED + lang + r[0] + r[1]).encode("utf-8")))
    chosen, count = [], Counter()
    seen_bytes, seen_text = set(), set()
    for sid, text in records:
        if sid in excluded_ids or len(chosen) >= CAP * len(options):
            continue
        t_hash = sha(text.encode("utf-8", "surrogatepass"))
        if t_hash in exclude_texts or t_hash in seen_text:
            continue
        salt = sha((TRAIN_SEED + lang + sid).encode())
        possible = [options[(int(salt[:8], 16) + i) % len(options)]
                    for i in range(len(options))]
        for label in possible:
            if label not in allowed or count[label] >= CAP:
                continue
            try:
                raw = text.encode(label, "strict")
                if raw.decode(label, "strict") != text:
                    continue
            except (UnicodeError, LookupError):
                continue
            # Must be distinct Unicode evidence against every OTHER codec
            # that is considered in this source language's comparison set.
            distinguishable = True
            for other in options:
                if other == label or other not in allowed:
                    continue
                try:
                    if raw.decode(other, "strict") == text:
                        distinguishable = False
                        break
                except (UnicodeError, LookupError):
                    pass
            if not distinguishable:
                continue
            key = sha(raw)
            if key in seen_bytes:
                continue
            seen_bytes.add(key)
            seen_text.add(t_hash)
            chosen.append((raw, label, f"cc0:{lang}:{sid}"))
            count[label] += 1
            break
    return chosen, dict(sorted(count.items()))


def cc0_training(heldout_ids):
    all_samples, source_info = [], {}
    for lang in SOURCES:
        url, blob = get_archive(lang)
        assert sha(blob) == EXPECTED[lang], "Tatoeba CC0 source drift: stop"
        reserved, quota, _ = select(blob, lang)
        excluded = {r["sentence_id"] for r in reserved}
        extra, counts = trainable_evidence(lang, set(SOURCES[lang]), blob, excluded, heldout_ids)
        all_samples += extra
        source_info[lang] = {
            "selected":len(extra), "by_label":counts,
            "reserved_confirmation_rows_not_used":len(excluded),
        }
    return all_samples, source_info


def summarize(rows):
    report = {"n":len(rows)}
    for mode in ("original","cc0_enriched"):
        report[mode] = {
            "correct_decoded_text":sum(r[mode]["text"] for r in rows),
            "correct_label":sum(r[mode]["label"] for r in rows),
            "abstain":sum(r[mode]["abstain"] for r in rows),
        }
    for metric in ("text","label"):
        h = sum(r["cc0_enriched"][metric] and not r["original"][metric] for r in rows)
        d = sum(r["original"][metric] and not r["cc0_enriched"][metric] for r in rows)
        report[metric + "_paired"] = {"helped":h,"harmed":d,"net":h-d}
    by_family = {}
    for fam in sorted({family(r["truth"]) for r in rows}):
        fam_rows = [r for r in rows if family(r["truth"]) == fam]
        by_family[fam] = {
            "n":len(fam_rows),
            "base_correct_text":sum(r["original"]["text"] for r in fam_rows),
            "enriched_correct_text":sum(r["cc0_enriched"]["text"] for r in fam_rows),
        }
    report["families"] = by_family
    return report


def run(args):
    rows = corpus(Path(args.fixtures))
    allocation, split = component_folds(rows)
    all_records, folds = [], []
    for fold in range(3):
        train = [(b, label, group) for (b, label, group, _, digest), k in zip(rows, allocation) if k != fold]
        held = [(b, label, group, digest) for (b, label, group, _, digest), k in zip(rows, allocation) if k == fold]
        heldids = {text_id(b,label) for b,label,group,digest in held}
        residual, removed = residual_subset(train)
        cc0, sources = cc0_training(heldids)
        # No same bytes conflicting labels: explicitly drop CC0 duplicates
        # already present in training rather than silently overwriting them.
        existing = {sha(b) for b,label,group in residual}
        extras = [x for x in cc0 if sha(x[0]) not in existing]
        assert all(text_id(data,label) not in heldids for data,label,group in extras)
        assert {g for _,_,g in extras}.isdisjoint({g for _,_,g,_ in held})
        originals = TrainableDetector.fit(residual, feature_profile="stable23")
        candidate = TrainableDetector.fit(residual+extras, feature_profile="stable23")
        reports = []
        for raw,truth,group,digest in held:
            truth_str = strict_text(raw,truth)
            assert truth_str is not None
            reports.append({
                "sha":digest,"group":group,"truth":truth,"fold":fold,
                "original":examine(originals,raw,truth,truth_str),
                "cc0_enriched":examine(candidate,raw,truth,truth_str),
            })
        all_records += reports
        folds.append({"fold":fold, "base_train":len(residual),
                      "cc0_train_added":len(extras),"source_counts":sources,
                      "heldout":summarize(reports)})
        print("CC0_ENRICH_FOLD_DONE",fold,json.dumps({
            "base_train":len(residual),"cc0_added":len(extras),
            "paired":folds[-1]["heldout"]["text_paired"],
        }),flush=True)
    report = {
        "schema":"codeccat-cc0-external-training-v1",
        "status":"development-only; no model promotion",
        "training_origin":"chardet/test-data + new CC0 text never selected in frozen CC0 confirmation",
        "validation_origin":"previously studied but exact-document-disjoint chardet/test-data",
        "method":"same stable23; baseline residual vs CC0 discriminative extra samples",
        "fold_split":split, "aggregated":summarize(all_records),
        "folds":folds,
        "limitations":[
            "Previously inspected development folds are not independent final acceptance",
            "Synthetic reencoded CC0 author sentences are not naturally observed codec prevalence",
            "Class/text ambiguity filtered only for within-source codec target lists",
            "Fresh real-byte, source-disjoint confirmation required prior to any release",
        ],
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"paired.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True) for x in all_records)+"\n")
    print("CC0_TRAIN_RESULT_BEGIN",flush=True)
    print(json.dumps({"aggregated":report["aggregated"],"folds":[{"fold":r["fold"],"cc0_train_added":r["cc0_train_added"],"source_counts":r["source_counts"]} for r in folds]},indent=2,sort_keys=True),flush=True)
    print("CC0_TRAIN_RESULT_END",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixtures",required=True)
    parser.add_argument("--out",default="research-results/cc0-training")
    run(parser.parse_args())
