"""Descriptive analysis of already-inspected sources using published Stable.

Not an independent holdout, model promotion gate or new source. Read pinned
training and external corpora ephemerally; write aggregate metadata only.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets" / "official"))
import audit_holdout_overlap as overlap
import gtk_blind_holdout as gtk
import uchardet_blind_holdout as uch

import chardet
import codeccat

EXPECTED_COMMIT_MANIFEST = "8ec3007dbd59336cee6018cfa44aeded39cca205e93f5fc9f24ee5ecf87f3ca9"
OUT = Path("research-results/stable-quality-triage.json")


def summarize_case_rows(rows: list[tuple[str, str, bool, bool, str]]) -> dict:
    """Each row: (bucket, encoding, Codeccat-correct, chardet-correct, outcome)."""
    grouped = defaultdict(list)
    for bucket, encoding, ours, competitor, outcome in rows:
        grouped[bucket].append((encoding, ours, competitor, outcome))
        grouped["all"].append((encoding, ours, competitor, outcome))
    result = {}
    for bucket, members in sorted(grouped.items()):
        per = defaultdict(lambda: {"n":0,"codeccat_correct":0,"chardet_correct":0,"codeccat_abstained":0})
        result[bucket] = {
            "n": len(members),
            "codeccat_correct": sum(item[1] for item in members),
            "chardet_correct": sum(item[2] for item in members),
            "codeccat_abstained": sum(item[4] == "abstained" for item in members),
        }
        for encoding, ours, other, outcome in members:
            metrics=per[encoding]
            metrics["n"] += 1
            metrics["codeccat_correct"] += int(ours)
            metrics["chardet_correct"] += int(other)
            metrics["codeccat_abstained"] += int(outcome=="abstained")
        result[bucket]["by_codec"] = dict(sorted(per.items()))
    return result


def outcome(runtime, raw: bytes, truth: str):
    result=runtime.detect(raw)
    correct=(result.text == truth)
    label=result.encoding
    return correct, "abstained" if label is None else ("correct" if correct else "incorrect")


def main() -> None:
    if importlib.metadata.version("codeccat") != "0.1.0":
        raise ValueError("must evaluate the published Stable 0.1.0 Runtime")
    train=Path("source-data/codeccat-replay-manifest.jsonl")
    if hashlib.sha256(train.read_bytes()).hexdigest() != EXPECTED_COMMIT_MANIFEST:
        raise ValueError("historical source manifest changed")

    screening=overlap.audit(train,Path("blind-source/po"),Path("external-encoding-source/test"))
    source_results=screening["source_results"]
    matched_uch={
        x["external_sha256"]: ("exact" if x["exact_full"] or x["exact_prefix"] else "candidate")
        for x in source_results["uchardet"]["flagged"]
    }
    matched_gtk={
        x["external_sha256"]: ("exact" if x["exact_full"] or x["exact_prefix"] else "candidate")
        for x in source_results["gtk"]["flagged"]
    }
    det=codeccat.CodecCat()
    gtk_rows=[]
    for row in gtk.cases():
        raw=row["bytes"]
        correct,status=outcome(det,raw,row["truth_text"])
        other_enc=gtk.norm(chardet.detect(raw).get("encoding"))
        other_text=gtk.decoded(raw,other_enc)
        bucket=matched_gtk.get(hashlib.sha256(raw).hexdigest(),"not_detected")
        gtk_rows.append((bucket,row["truth_label"],correct,other_text==row["truth_text"],status))
    uch_rows=[]
    source_rows,skips=uch.source_samples()
    for row in source_rows:
        raw=row["raw"]
        correct,status=outcome(det,raw,row["expected"])
        other_enc=uch.norm(chardet.detect(raw).get("encoding"))
        other_text=uch.text(raw,other_enc)
        bucket=matched_uch.get(hashlib.sha256(raw).hexdigest(),"not_detected")
        uch_rows.append((bucket,row["truth"],correct,other_text==row["expected"],status))
    if len(gtk_rows)!=328 or len(uch_rows)!=102:
        raise ValueError(f"historical external selection changed: gtk {len(gtk_rows)}, uchardet {len(uch_rows)}")
    def counts(rows):
        return Counter(bucket for bucket, *_ in rows)
    for label,rows,expected in (
        ("gtk",gtk_rows,source_results["gtk"]),
        ("uchardet",uch_rows,source_results["uchardet"]),
    ):
        buckets=counts(rows)
        if buckets["exact"] != expected["exact_full_matches"]:
            raise ValueError(f"{label}: exact overlap join mismatch")
        if buckets["exact"]+buckets["candidate"] != expected["candidate_near_overlap_windows"]:
            raise ValueError(f"{label}: candidate overlap join mismatch")
    summary={
        "schema":"codeccat-existing-corpus-failure-triage-v1",
        "runtime_version":"0.1.0",
        "chardet_version":importlib.metadata.version("chardet"),
        "source_commit_pins":{
            "training":"b0c0d206913ba118ffea2f7d306883e7a4a76db7",
            "gtk":"7c578d79eb8347b14a1717bfcc0084674a43c562",
            "uchardet":"4e685757780cb3c652fc6c9ec759f62888969ec9",
        },
        "gtk":summarize_case_rows(gtk_rows),
        "uchardet":summarize_case_rows(uch_rows),
        "uchardet_excluded_unsupported_or_invalid":len(skips),
        "training_overlap_screen":{
            name:{k:v for k,v in records.items() if k!="flagged"}
            for name,records in source_results.items()
        },
        "interpretation":"These corpora were inspected previously. Exact matches invalidate any claim of a pristine 102-case uchardet holdout. Candidate overlaps require review. Unflagged windows are not proven independent; no promotion decision can use this alone.",
        "stores_raw_text":False,
        "previously_observed_data":True,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    compact={
        k:{bucket:{field:val for field,val in stats.items() if field!="by_codec"}
           for bucket,stats in summary[k].items()}
        for k in ("gtk","uchardet")
    }
    print("STABLE_QUALITY_FAILURE_TRIAGE",json.dumps(compact,sort_keys=True))


if __name__ == "__main__":
    main()
