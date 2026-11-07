"""P1 dual-oracle research: Python strict codec vs WHATWG TextDecoder.

Use unchanged published CodecCat 0.1.1. This is an *evaluation* not a change
to the runtime, Unicode output or training. HTML source data remain ephemeral.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import base64

import codeccat

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "native-prototype"))
from evaluate_c1_real_html import ARCHIVE, prepare, train_hashes, truth_from_label
from research_p1_html_encoding import declared

OUT=Path("research-results/p1-whatwg-dual-oracle.json")
NODE=Path("research/scripts/p1_whatwg_decoder.js")
MAX_BATCH_BYTES=8*1024*1024


def hash_text(text:str|None)->str|None:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None


def node_batch(items:list[dict])->list[dict]:
    p=subprocess.run(
        ["node",str(NODE)],input=json.dumps(items,separators=(",",":")),
        text=True,capture_output=True,check=True,timeout=120,
    )
    out=json.loads(p.stdout)
    assert [x["id"] for x in out] == [x["id"] for x in items]
    return out


def main():
    assert importlib.metadata.version("codeccat")=="0.1.1"
    train=train_hashes()
    detector=codeccat.CodecCat()
    rows=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        eligible,exclusions=prepare(z,train)
        assert len(eligible)==1886,len(eligible)
        for i,(path,group,codec,sha,nbytes) in enumerate(eligible):
            raw=z.read(path)
            assert hashlib.sha256(raw).hexdigest()==sha
            detection=detector.detect(raw)
            py=truth_from_label(raw,codec)
            h=declared(raw)
            labels={"archive":codec}
            if h:
                labels["early_meta"]=h
            rows.append({
                "group":group,"path_sha256":hashlib.sha256(path.encode()).hexdigest(),
                "sha256":sha,"bytes":nbytes,"raw":raw,"py_hash":hash_text(py),
                "detector_hash":hash_text(detection.text),
                "predicted":detection.encoding,
                "candidates":list(detection.candidates),
                "labels":labels,
                "has_early_meta":bool(h),
            })
    summaries=defaultdict(Counter)
    predictions=defaultdict(Counter)
    examples=defaultdict(list)
    pending=[]
    bytes_pending=0
    total=0

    def assess(batch,results):
        nonlocal total
        for row, web in zip(batch,results):
            total+=1
            s=summaries[row["group"]]
            s["n"]+=1
            archive=web["checks"]["archive"]
            py_match=row["detector_hash"]==row["py_hash"]
            s["baseline_matches_python_directory"]+=int(py_match)
            s["baseline_misses_python_directory"]+=int(not py_match)
            s["whatwg_directory_strict_valid"]+=int(archive["valid"])
            if not archive["valid"]:
                s["whatwg_directory_invalid"]+=1
            else:
                webh=archive["sha256"]
                eq=row["detector_hash"]==webh
                python_web_different=row["py_hash"]!=webh
                s["python_vs_whatwg_mapping_differs"]+=int(python_web_different)
                s["baseline_matches_whatwg_directory"]+=int(eq)
                s["baseline_misses_whatwg_directory"]+=int(not eq)
                s["python_miss_but_web_match"]+=int(not py_match and eq)
                s["python_match_but_web_miss"]+=int(py_match and not eq)
                if not eq:
                    predictions[row["group"]][str(row["predicted"])]+=1
                # This is a *potential* upper bound, not proof of a better
                # predictive codec: WHATWG uses labels from the reference.
                if row["has_early_meta"]:
                    s["meta_present_web_valid"]+=int(web["checks"]["early_meta"]["valid"])
                    m=web["checks"]["early_meta"]
                    if m["valid"]:
                        s["early_meta_maps_to_web_archive_encoding"]+=int(m["encoding"]==archive["encoding"])
                        s["meta_unicode_hash_matches_web_archive"]+=int(m["sha256"]==webh)
                        s["meta_web_changes_baseline_bad_to_reference"]+=int(not eq and m["sha256"]==webh)
                        s["meta_web_changes_baseline_good_to_bad"]+=int(eq and m["sha256"]!=webh)
                    elif len(examples[row["group"]])<10:
                        examples[row["group"]].append({
                            "sha256":row["sha256"],"predicted":row["predicted"],
                            "case":"early-meta-label-invalid-for-WHATWG",
                            "meta_label":row["labels"].get("early_meta")})
                # Classify whether any existing candidate's Python strict
                # decode matches the browser reference, without reranking.
                # Decode only inside ephemeral CI; export no content.
                try:
                    import codecs
                    raw=row["raw"]
                    candidate_hits=[]
                    for label in row["candidates"]:
                        try:
                            cand=raw.decode(label,"strict")
                            if hash_text(cand)==webh:
                                candidate_hits.append(label)
                        except (LookupError,UnicodeError):
                            pass
                    if not eq:
                        s["whatwg_misses_recoverable_by_python_candidate"]+=int(bool(candidate_hits))
                        s["whatwg_misses_not_recoverable_by_candidates"]+=int(not candidate_hits)
                    if len(examples[row["group"]])<12 and (python_web_different or (not eq and candidate_hits)):
                        examples[row["group"]].append({
                            "sha256":row["sha256"],"bytes":row["bytes"],
                            "case":"mapping-difference" if python_web_different else "candidate-could-recover",
                            "predicted":row["predicted"],
                            "candidates":row["candidates"],
                            "matching_candidate_encodings":candidate_hits,
                            "python_matches":py_match,"web_matches":eq,
                            "web_label":archive["encoding"],
                            "has_early_meta":row["has_early_meta"]
                        })
                except Exception:
                    raise

    for i,row in enumerate(rows):
        item={"id":i,"data":base64.b64encode(row["raw"]).decode("ascii"),"labels":row["labels"]}
        if pending and bytes_pending+len(item["data"])>MAX_BATCH_BYTES:
            result=node_batch(pending)
            assess(rows[i-len(pending):i],result)
            pending=[]
            bytes_pending=0
        pending.append(item)
        bytes_pending+=len(item["data"])
    if pending:
        result=node_batch(pending)
        assess(rows[len(rows)-len(pending):],result)
    assert total==len(rows)==1886
    report={
        "schema":"codeccat-p1-python-whatwg-dual-oracle-v1",
        "runtime_version":"0.1.1",
        "node_textdecoder_semantics":"ICU-backed WHATWG-compatible TextDecoder(fatal=True); replacement is not allowed; default BOM processing",
        "method":"original bytes; full/prefix/sampled-block historic training overlap screen; strict Python directory label vs independent Node TextDecoder directory label",
        "groups":{k:dict(v) for k,v in sorted(summaries.items())},
        "when_web_oracle_missed_predicted_encodings":{k:dict(v.most_common(15)) for k,v in sorted(predictions.items())},
        "example_metadata_only":dict(examples),
        "exclusions":{k:dict(v) for k,v in exclusions.items()},
        "caveats":[
            "Archive directory labels are a proxy, not human-confirmed textual truth.",
            "WHATWG browser decoding differs from Python strict; strict TextDecoder may reject legacy HTML accepted by real browsers using replacement.",
            "Early meta is a regex first-1024-byte heuristic, not compliant HTML prescan or HTTP Content-Type authority.",
            "Comparing archive label to meta is not independent validation when both come from the same crawled source.",
            "Some cases may have nonstandard C1 controls or multiple legitimate byte interpretations.",
            "No codeccat model, default inference, candidate ranking, API or release modified."
        ],
        "raw_html_exported":False,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("P1_DUAL_ORACLE_SUMMARY",json.dumps(report["groups"],sort_keys=True))
    print("P1_DUAL_ORACLE_MISSES",json.dumps(report["when_web_oracle_missed_predicted_encodings"],sort_keys=True))


if __name__=="__main__":
    main()
