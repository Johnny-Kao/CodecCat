"""Attach research-only prototype results to the existing screened browser corpus.

Run AFTER p1_prepare_browser_corpus.py; private manifest and raw files never
uploaded. Only the subsequent browser experiment writes sanitized aggregates.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path("research/prototypes").resolve()))
from html_aware import decode_html_prototype

PRIVATE=Path("private-browser-fixtures")
MANIFEST=PRIVATE/"manifest.json"


def main():
    entries=json.loads(MANIFEST.read_text(encoding="utf-8"))
    durations=[]
    for entry in entries:
        data=(PRIVATE/entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest()==entry["sha256"]
        start=time.perf_counter_ns()
        result=decode_html_prototype(data)
        elapsed=(time.perf_counter_ns()-start)/1000
        durations.append(elapsed)
        entry["prototype_encoding"]=result.encoding
        entry["prototype_source"]=result.source
        entry["prototype_text_sha256"]=hashlib.sha256(result.text.encode("utf-8")).hexdigest()
        entry["prototype_replacements"]=result.used_replacement
    MANIFEST.write_text(json.dumps(entries,separators=(",",":")),encoding="utf-8")
    print("P1_HTML_PROTOTYPE_PREPARED",json.dumps({
        "n":len(entries),
        "latency_us_p50":round(statistics.median(durations),1),
        "latency_us_p95":round(sorted(durations)[int((len(durations)-1)*0.95)],1),
        "source_counts":{s:sum(x["prototype_source"]==s for x in entries)
                         for s in ("bom","override","transport","meta","fallback")},
        "raw_html_exported":False
    },sort_keys=True))


if __name__=="__main__":
    main()
