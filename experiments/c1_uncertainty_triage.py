"""Assess non-destructive C1 uncertainty signaling on the exact 363-case corpus.

No result text or encoding is modified. This is a research-only wrapper to
quantify which correctly/incorrectly decoded samples would be flagged.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import codeccat

from western_c1_counterfactual import rerank_control

WORK = Path(".overall-work")
OUT = Path("research-results/c1-uncertainty-triage.json")


def eligible_flag(raw, result):
    if result.text is None or not any(0x80 <= ord(c) <= 0x9f for c in result.text):
        return False
    for candidate in result.candidates:
        if candidate == result.encoding:
            continue
        try:
            alternative = raw.decode(candidate, "strict")
        except (LookupError, UnicodeError):
            continue
        if not any(0x80 <= ord(c) <= 0x9f for c in alternative):
            return True
    return False


def main():
    assert importlib.metadata.version("codeccat")=="0.1.0"
    det = codeccat.CodecCat()
    rows = json.loads((WORK/"manifest.json").read_text())
    by_pool=defaultdict(Counter)
    by_codec=defaultdict(Counter)
    for row in rows:
        data=(WORK/row["file"]).read_bytes()
        if row["pool"]=="natural_utf8":
            truth=data.decode("utf-8")
        elif row["pool"]=="cjk_natural_preexamined":
            name=row["source"].removesuffix(".txt")
            truth=(Path("external-cpython/Lib/test/cjkencodings") /
                   (name+"-utf8.txt")).read_bytes().decode("utf-8")
        else:
            truth=data.decode(row["encoding"],"strict")
        assert hashlib.sha256(truth.encode("utf-8")).hexdigest()==row["truth_sha256"]
        result=det.detect(data)
        flagged=eligible_flag(data,result)
        baseline_ok=result.text==truth
        for table in (by_pool[row["pool"]],by_codec[row["encoding"]]):
            table["n"]+=1
            table["wrong"]+=int(not baseline_ok)
            table["flagged"]+=int(flagged)
            table["flagged_wrong"]+=int(flagged and not baseline_ok)
            table["flagged_correct"]+=int(flagged and baseline_ok)
            table["unflagged_wrong"]+=int(not flagged and not baseline_ok)
            table["flagged_previously_ambiguous"]+=int(flagged and result.ambiguous)
    report={
        "schema":"codeccat-c1-nondestructive-uncertainty-v1",
        "status":"analysis_only",
        "by_pool":{k:dict(v) for k,v in sorted(by_pool.items())},
        "by_codec":{k:dict(v) for k,v in sorted(by_codec.items())},
        "raw_content_exported":False,
        "tradeoff":"Flagging is not improved correctness: it preserves the same erroneous decoded text, while notifying callers if an alternative strict decoder has no C1 controls.",
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("C1_NONDESTRUCTIVE_TRIAGE",json.dumps(report["by_pool"],sort_keys=True))


if __name__=="__main__":
    main()
