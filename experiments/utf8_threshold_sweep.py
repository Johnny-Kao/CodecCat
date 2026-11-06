"""Diagnostic only: sweep UTF-8 evidence thresholds against *published* a4.

Records helps/harms before any code change. The two sources have already
been inspected and are development diagnostics, NOT untouched holdouts.
"""
from __future__ import annotations
import json,sys,collections
from pathlib import Path
sys.path.insert(0,"experiments")
import gtk_blind_holdout as gtk
import uchardet_blind_holdout as multi
from codeccat import CodecCat
import codeccat.vnext_decoding as decoding

cases=[]
for x in gtk.cases():
    cases.append(dict(pool="gtk",raw=x["bytes"],truth=x["truth_text"],label=x["truth_label"],source=x["file"]))
ms,_=multi.source_samples()
for x in ms:
    cases.append(dict(pool="multicodec",raw=x["raw"],truth=x["expected"],label=x["truth"],source=x["source"]))
print("SOURCE_COUNTS",dict(collections.Counter(x["pool"] for x in cases)),flush=True)

def evidence(raw,threshold):
    from codeccat.vnext_bom import inspect_bom
    if len(raw)<64 or inspect_bom(raw).signature is not None:return False
    try: txt=raw.decode("utf-8","strict")
    except UnicodeError:return False
    if any((ord(ch)<32 and ch not in "\t\n\r") or 0x7f<=ord(ch)<0xa0 for ch in txt):return False
    return sum(ord(ch)>=128 for ch in txt)>=threshold

det=CodecCat()
saved=decoding._strong_utf8_evidence
try:
    decoding._strong_utf8_evidence=saved
    original=[det.detect(x["raw"]).text==x["truth"] for x in cases]
    output=[]
    for threshold in (1,2,3,4,5,6,8,10,12,16):
        decoding._strong_utf8_evidence=lambda raw,t=threshold:evidence(raw,t)
        new=[det.detect(x["raw"]).text==x["truth"] for x in cases]
        per={}
        for group in ("gtk","multicodec"):
            indices=[i for i,x in enumerate(cases) if x["pool"]==group]
            helps=[i for i in indices if not original[i] and new[i]]
            harms=[i for i in indices if original[i] and not new[i]]
            per[group]=dict(n=len(indices),old=sum(original[i] for i in indices),
                new=sum(new[i] for i in indices),helps=len(helps),harms=len(harms),
                harm_labels=dict(collections.Counter(cases[i]["label"] for i in harms)),
                harm_sources=[cases[i]["source"] for i in harms][:25],
                gain_labels=dict(collections.Counter(cases[i]["label"] for i in helps)))
        output.append(dict(threshold=threshold,per=per))
    Path("research-results/utf8-sweep").mkdir(parents=True,exist_ok=True)
    Path("research-results/utf8-sweep/summary.json").write_text(json.dumps(output,indent=2)+"\n")
    print("UTF8_THRESHOLD_SWEEP",json.dumps(output),flush=True)
finally:
    decoding._strong_utf8_evidence=saved
