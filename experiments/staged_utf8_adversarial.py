"""Paired staging evaluation. The original benchmark cases are development data.

Also evaluate intentionally ambiguous legacy-encoding inputs as negative
controls: strict UTF-8 validity is NOT proof of original encoding.
"""
import base64,collections,json,sys
from pathlib import Path
sys.path.insert(0,"experiments")
import gtk_blind_holdout as gtk
import uchardet_blind_holdout as multi
import five_corpus_benchmark as five
import native_utf8_external_gate as glib
from codeccat import CodecCat
import codeccat.vnext_decoding as d

source=[]
for x in gtk.cases():
    source.append(("gtk",x["bytes"],x["truth_text"],x["truth_label"]))
m,_=multi.source_samples()
for x in m:source.append(("multicodec",x["raw"],x["expected"],x["truth"]))
five.build_fixtures()
for n in range(1,6):
    for x in json.loads((five.OUT/f"corpus-{n}.json").read_text()):
        raw=base64.b64decode(x["data"])
        source.append(("five_corpus",raw,five.get_text(raw,x["encoding"]),x["encoding"]))
for x in glib.cases():source.append(("glib",x["raw"],x["expected"],x["truth"]))

def adversarial():
    # These valid legacy-encoded bytes *also* parse as UTF-8. Neither
    # encoding can be universally identified from bytes alone.
    out=[]
    for codec in ("cp1252","iso8859-1"):
        for count in (1,2,3,4,6,8):
            for glyph in ("Ã©","Ã¶","Â£","Ã¨"):
                phrase=("English ASCII header and text line. " * 8 + glyph*count)
                raw=phrase.encode(codec)
                out.append((f"ambiguous:{codec}",raw,phrase,codec))
    return out

source+=adversarial()
det=CodecCat()
original=d._strong_utf8_evidence
def floor(raw,n):
    from codeccat.vnext_bom import inspect_bom
    if len(raw)<64 or inspect_bom(raw).signature:return False
    try: txt=raw.decode("utf-8","strict")
    except UnicodeError:return False
    if any((ord(ch)<32 and ch not in "\t\n\r") or (127<=ord(ch)<160) for ch in txt):return False
    return sum(ord(ch)>=128 for ch in txt)>=n

def scoring():
    out=[]
    for pool,raw,truth,label in source:
        r=det.detect(raw)
        out.append(r.text==truth)
    return out

try:
    d._strong_utf8_evidence=lambda raw:floor(raw,16)
    old=scoring()
    d._strong_utf8_evidence=original
    new=scoring()
finally:
    d._strong_utf8_evidence=original

by={}
for pool in sorted(set(row[0] for row in source)):
    index=[i for i,r in enumerate(source) if r[0]==pool]
    helps=[i for i in index if new[i] and not old[i]]
    harms=[i for i in index if old[i] and not new[i]]
    by[pool]={"n":len(index),"old":sum(old[i] for i in index),"new":sum(new[i] for i in index),
              "helps":len(helps),"harms":len(harms)}
print("UTF8_STAGE_PAIRED",json.dumps(by,sort_keys=True),flush=True)
Path("research-results/utf8-stage").mkdir(parents=True,exist_ok=True)
Path("research-results/utf8-stage/summary.json").write_text(json.dumps(by,indent=2)+"\n")
assert sum(by[k]["harms"] for k in ("gtk","multicodec","five_corpus","glib")) == 0, "regression in known fixtures"
assert by["gtk"]["new"]>=320
if by["ambiguous:cp1252"]["harms"] or by["ambiguous:iso8859-1"]["harms"]:
    print("ADVERSARIAL_AMBIGUITY_REGRESSION",json.dumps({k:by[k] for k in by if k.startswith("ambiguous:")}),flush=True)
