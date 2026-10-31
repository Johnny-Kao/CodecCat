"""Assess likely stale legacy codec labels in the existing pinned training corpus.

NO MODEL FIT OR SCORING. Data is from the already-consumed development
repository. Findings require human review; do not relabel or delete samples
without independent source provenance.
"""
from __future__ import annotations
import argparse,collections,json,pathlib,subprocess,sys

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_custom_retrain_gate import suspicious_singlebyte_utf8

TRAIN_REV="b0c0d206913ba118ffea2f7d306883e7a4a76db7"


def run(args):
    root=pathlib.Path(args.fixtures)
    sha=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if sha!=TRAIN_REV: raise RuntimeError("not pinned training corpus")
    cases=corpus(root)
    suspect=[];by_label=collections.Counter();all_labels=collections.Counter()
    for raw,label,group,fold,digest in cases:
        all_labels[label]+=1
        issue=suspicious_singlebyte_utf8(raw,label)
        if issue is not None:
            by_label[label]+=1
            suspect.append({"byte_sha":digest,"source_group":group,
                            "declared_label":label,"original_fold":fold,**issue})
    report={
        "schema":"codeccat-original-training-label-integrity-audit-v1",
        "role":"SOURCE_LABEL_AUDIT_ONLY_NO_RETRAIN",
        "source_commit":TRAIN_REV,
        "n_rows":len(cases),
        "labels_all":dict(sorted(all_labels.items())),
        "suspected_count":len(suspect),
        "suspected_by_label":dict(sorted(by_label.items())),
        "evidence":suspect,
        "limits":[
            "Source labels can still be correct if text intentionally contains mojibake",
            "4096-byte research prefixes may truncate evidence and hide some matches",
            "No training, fixture editing, or retrospective exclusion is performed",
        ],
    }
    output=pathlib.Path(args.out);output.mkdir(parents=True,exist_ok=True)
    (output/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("PINNED_TRAINING_LABEL_AUDIT_START")
    print(json.dumps({"source_commit":sha,"rows":len(cases),"suspected_count":len(suspect),
                      "suspected_by_label":dict(sorted(by_label.items())),
                      "examples":suspect[:12]},sort_keys=True,indent=2),flush=True)
    print("PINNED_TRAINING_LABEL_AUDIT_END")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fixtures",required=True)
    p.add_argument("--out",default="research-results/training-label-audit")
    run(p.parse_args())
