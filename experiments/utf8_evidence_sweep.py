"""Counterfactual analysis for conservative UTF-8 evidence.

Use exact release wheel, frozen five-corpus samples, no tuning on an
unseen held-out source. Diagnostic only: never ship a proposed rule merely
because it performs well on these studied samples.
"""
from __future__ import annotations
import base64
import json
from collections import Counter
from pathlib import Path
import codeccat

ROOT=Path("research-results/five-corpus")
THRESHOLDS=(8,16,24,32,48,64,96,128,192)
detector=codeccat.CodecCat()

def utf8_evidence(b):
    try:
        text=b.decode("utf-8","strict")
    except UnicodeError:
        return None
    if any((ord(c)<32 and c not in "\t\r\n") or (0x7f <=ord(c)<0xa0) for c in text):
        return None
    return sum(ord(c)>127 for c in text)

rows=[]
for k in range(1,6):
    for r in json.loads((ROOT/f"corpus-{k}.json").read_text()):
        b=base64.b64decode(r["data"])
        d=detector.detect(b)
        true=b.decode(r["encoding"],"strict").lstrip("\ufeff")
        n=utf8_evidence(b)
        rows.append(dict(pool=r["pool"],encoding=r["encoding"],sha256=r["sha256"],
                         current=d.text==true,changed=d.encoding!="utf-8",
                         chars=n,txt=d.text,expected=true,raw=b))
results={}
for t in THRESHOLDS:
    outcome={"threshold":t,"n":len(rows),"invoked":0,"helps":0,"harms":0,"by_pool":{}}
    for pool in ("chardet","charset_normalizer","codeccat_external"):
        subset=[r for r in rows if r["pool"]==pool]
        changes=[r for r in subset if r["changed"] and r["chars"] is not None and r["chars"]>=t]
        helps=sum(not r["current"] and r["raw"].decode("utf-8")==r["expected"] for r in changes)
        harms=sum(r["current"] and r["raw"].decode("utf-8")!=r["expected"] for r in changes)
        outcome["by_pool"][pool]={"n":len(subset),"invoked":len(changes),
                                   "helps":helps,"harms":harms}
        outcome["invoked"]+=len(changes)
        outcome["helps"]+=helps
        outcome["harms"]+=harms
    results[str(t)]=outcome
report={"model":"published Native v0.1.0a3","n":len(rows),"thresholds":results,
        "warning":"Candidate evaluated on known development and previously inspected sources, not a fresh blind acceptance source. This is not an authorized product change."}
(ROOT/"utf8-counterfactual.json").write_text(json.dumps(report,indent=2)+"\n")
print("UTF8_EVIDENCE_SWEEP",json.dumps(report),flush=True)
