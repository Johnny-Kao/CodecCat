"""Investigate already available trainable language-coherence L3 against native legacy bytes.

DEVELOPMENT ABALATION ON ALREADY CONSUMED EXTERNAL HOLDOUT!
This script is *not* another new blind confirmation, and does not update a
production model. It tests whether the existing optional L3 is a maintainable
alternative to introducing language/codec-specific heuristics.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json

from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_selective import SelectiveLatinDetector
from codeccat.vnext_utf8_evidence import Utf8EvidenceDetector
from codeccat.vnext_l3_language import LanguageCoherence
from codeccat.vnext_decoding import decode

from vnext_legacy_po_one_shot import (
    pin, frozen_rows, TRAIN_REV, SOURCES, FROZEN_FINGERPRINT, sha,
)
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_content_safe_contrast import contrastive_augmentation
from vnext_training_strategy_audit import strict_text


def compare(rows):
    modes=("stable23","stable23_l3","selective_l3","utf8_l3")
    ret={"n":len(rows),"score":{},"paired":{},"by_codec":{},"by_source":{}}
    for mode in modes:
        vals=[row["scores"][mode] for row in rows]
        ret["score"][mode]={
            "correct_text":sum(x["correct_text"] for x in vals),
            "l3_called":sum(x["l3_called"] for x in vals),
            "l3_changed":sum(x["l3_changed"] for x in vals),
        }
    for mode in modes[1:]:
        h=sum(row["scores"][mode]["correct_text"] and not row["scores"]["stable23"]["correct_text"]
              for row in rows)
        d=sum(row["scores"]["stable23"]["correct_text"] and not row["scores"][mode]["correct_text"]
              for row in rows)
        ret["paired"][mode]={"helps":h,"harms":d,"net":h-d}
    for attr,key in (("source","by_source"),("codec","by_codec")):
        for v in sorted({row[attr] for row in rows}):
            group=[row for row in rows if row[attr]==v]
            ret[key][v]={"n":len(group),**{
                mode:sum(row["scores"][mode]["correct_text"] for row in group)
                for mode in modes
            }}
    return ret


def run(args):
    roots={"lynx":Path(args.lynx).resolve(),"gettext":Path(args.gettext).resolve()}
    pin(Path(args.training),TRAIN_REV)
    rows=frozen_rows(args.manifest,roots)
    samples=[(b,y,g) for b,y,g,k,h in corpus(Path(args.training))]
    text_ids={sha(strict_text(b,y).encode("utf-8","surrogatepass")) for b,y,g in samples}
    seen=set();held=[]
    for row in rows:
        tid=sha(row["text"].encode("utf-8","surrogatepass"))
        if tid in text_ids or row["sha256"] in seen:
            continue
        seen.add(row["sha256"])
        held.append(row)
    if len(held)<100:
        raise RuntimeError("previous external diagnostic dataset changed")
    train,_=residual_subset(samples)
    augmented,_=contrastive_augmentation(train,per_direction=24,exclude_ebcdic=True)
    baseline=TrainableDetector.fit(train,feature_profile="stable23")
    latin=TrainableDetector.fit(train+augmented,feature_profile="stable23")
    coherent=LanguageCoherence.fit(samples)
    models={
        "stable23":(baseline,None),
        "stable23_l3":(baseline,coherent),
        "selective_l3":(SelectiveLatinDetector(baseline,latin),coherent),
        "utf8_l3":(Utf8EvidenceDetector(baseline,min_codepoints=16),coherent),
    }
    evaluated=[]
    for row in held:
        results={}
        for mode,(model,language) in models.items():
            result=decode(model,row["data"],bom_policy="preserve",language_model=language)
            results[mode]={
                "codec":result.encoding,"correct_text":result.text==row["text"],
                "l3_called":result.l3_called,"l3_changed":result.l3_changed,
            }
        evaluated.append({
            "source":row["source"],"file":row["file"],"window":row["window"],
            "codec":row["codec"],"sha256":row["sha256"],"scores":results,
        })
    summary=compare(evaluated)
    report={
        "schema":"codeccat-l3-native-legacy-development-ablation-v1",
        "source":"Lynx and GNU gettext frozen original bytes, consumed source",
        "source_freeze_sha":FROZEN_FINGERPRINT,
        "status":"DEVELOPMENT_REUSE; NOT_FRESH_HOLDOUT; NO_PROMOTION",
        "training":"preexisting chardet/test-data; all source-actual bytes excluded from training",
        "model":"existing opt-in LanguageCoherence trigram L3; unchanged default thresholds",
        "result":summary,
        "warnings":[
            "This test suite has previously been inspected in one-shot #41240514666; any improvement here is only a hypothesis",
            "Source windows are correlated within catalogs",
            "No new native raw bytes committed or used for fitting",
            "L3 activation adds inference cost, not measured here",
        ],
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    (out/"cases.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True) for x in evaluated)+"\n")
    print("L3_LEGACY_ABLATION_START",flush=True)
    print(json.dumps(summary,indent=2,sort_keys=True),flush=True)
    print("L3_LEGACY_ABLATION_END",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--lynx",required=True)
    p.add_argument("--gettext",required=True)
    p.add_argument("--training",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",default="research-results/legacy-l3-ablation")
    run(p.parse_args())
