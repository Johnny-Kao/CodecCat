"""Consumed-source regression replay, NOT a new external confirmation.

The CC0 source was used already in one-shot run 41233855912. This script
exists ONLY to prevent unacceptable regressions in the selective wrapper.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_decoding import decode
from codeccat.vnext_selective import SelectiveLatinDetector

sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_content_safe_contrast import contrastive_augmentation
from vnext_cc0_external_confirmation import (
    EXPECTED_TRAINING, prepared_rows,
)


def run(args):
    sha=subprocess.check_output(["git","-C",args.fixtures,"rev-parse","HEAD"],text=True).strip()
    if sha!=EXPECTED_TRAINING:
        raise RuntimeError("training revision differs")
    train=[(b,y,g) for b,y,g,fold,digest in corpus(Path(args.fixtures))]
    cases, sources, leakage=prepared_rows(train)
    base_train,_=residual_subset(train)
    generated,info=contrastive_augmentation(base_train,per_direction=24,exclude_ebcdic=True)
    baseline=TrainableDetector.fit(base_train)
    specialist=TrainableDetector.fit(base_train+generated)
    routed=SelectiveLatinDetector(baseline,specialist)
    rows=[]
    for case in cases:
        b=decode(baseline,case["data"],bom_policy="strip")
        s=decode(specialist,case["data"],bom_policy="strip")
        r=decode(routed,case["data"],bom_policy="strip")
        truth=case["text"]
        rows.append({
            "language":case["lang"], "true_codec":case["truth"],
            "baseline":b.text==truth, "specialist":s.text==truth, "routed":r.text==truth,
            "original_label":b.encoding,"specialist_label":s.encoding,"routed_label":r.encoding,
        })
    assert len(rows)>=450
    summary={"n":len(rows)}
    for name in ("baseline","specialist","routed"):
        summary[name]=sum(x[name] for x in rows)
    for name in ("specialist","routed"):
        helped=sum(x[name] and not x["baseline"] for x in rows)
        harmed=sum(x["baseline"] and not x[name] for x in rows)
        summary[name+"_paired"]={"helped":helped,"harmed":harmed,"net":helped-harmed}
    grouped=defaultdict(list)
    for row in rows:
        grouped[row["language"],row["true_codec"]].append(row)
    groups={}
    for (lang,code), subset in sorted(grouped.items()):
        groups[f"{lang}:{code}"]={
            "n":len(subset),
            **{k:sum(z[k] for z in subset) for k in ("baseline","specialist","routed")},
            "changes_due_to_routing":sum(z["original_label"]!=z["routed_label"] for z in subset),
        }
    # Contract: unrelated current baseline winners must NEVER be rerouted.
    disallowed=0
    for case in rows:
        x=case["original_label"] or ""
        if not (x.startswith("iso8859-") or x.startswith("cp12")):
            disallowed+=(case["original_label"]!=case["routed_label"])
    if disallowed:
        raise AssertionError(f"specialist changed {disallowed} baseline non-Latin decisions")
    report={
        "schema":"cc0-consumed-replay-v1",
        "status":"CONSUMED_EXTERNAL_SOURCE_REGRESSION_ONLY",
        "comparison":summary,
        "by_language_codec":groups,
        "nonlatin_baseline_changed":disallowed,
        "frozen_cc0_source":"https://github.com/Johnny-Kao/CodecCat/actions/runs/41233758720",
        "caveat":"This source was inspected in #41233855912; do not describe this as blind confirmation",
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("SELECTIVE_CC0_REPLAY_START")
    print(json.dumps(report,indent=2,sort_keys=True))
    print("SELECTIVE_CC0_REPLAY_END")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fixtures",required=True)
    p.add_argument("--out",default="research-results/cc0-routed-replay")
    run(p.parse_args())
