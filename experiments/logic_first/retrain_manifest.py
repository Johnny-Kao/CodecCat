"""Research retraining CLI: corpus manifest -> fit -> candidate -> fail-closed report.

Does not approve or activate. All paths are local on the CI runner.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from external_corpus_gate import validate
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.model_lifecycle import stage

def load_rows(manifest,root):
    data=defaultdict(list)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        record=json.loads(line)
        raw=(root/record["path"]).read_bytes()
        data[record["split"]].append((raw,record["encoding"],record["source_id"]))
    return data

def execute(manifest,root,incumbent_path,output):
    preflight=validate(manifest,root)
    if not preflight["ok"]:
        raise ValueError("external corpus invalid: "+"; ".join(preflight["errors"]))
    data=load_rows(manifest,root)
    if not data["train"] or not data["validation"] or not data["confirmation"]:
        raise ValueError("train, validation and untouched confirmation partitions are mandatory")
    incumbent=TrainableDetector.load_portable(incumbent_path)
    candidate=TrainableDetector.fit(data["train"])
    provenance={"source":str(manifest.name),"license":"per-record; see manifest",
        "manifest_sha256":preflight["manifest_sha256"],"origin":"user-provided provenance claims"}
    output=Path(output)
    result=stage(output,candidate,train_rows=data["train"],heldout_rows=data["validation"],
        provenance=provenance,incumbent=incumbent,min_samples=20)
    report={"schema":"codeccat-retrain-pipeline-v1","manifest":preflight,
            "candidate_sha256":result["model_sha256"],"gate":result["gate"],
            "confirmation_count_untouched":len(data["confirmation"]),
            "active_changed":False,"release_ready":False}
    (output/"run-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--manifest",type=Path,required=True)
 p.add_argument("--root",type=Path,required=True)
 p.add_argument("--incumbent",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 args=p.parse_args()
 report=execute(args.manifest,args.root,args.incumbent,args.output)
 print("RETRAIN_RESULT_START");print(json.dumps(report,indent=2));print("RETRAIN_RESULT_END")
 raise SystemExit(0 if report["gate"]["accepted"] else 2)
if __name__=="__main__":main()
