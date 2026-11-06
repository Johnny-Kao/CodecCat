"""Blind (before evaluation) third-party GLib PO source validation.

Immutable third-party commit is fixed before candidate scoring.
Records only model outcomes + input fingerprints, never uploads full text.
"""
from __future__ import annotations
import argparse,codecs,hashlib,json,re
from pathlib import Path
import codeccat

SOURCE=Path("benchmark-sources/glib/po")
OUT=Path("research-results/five-corpus")
def canon(enc):
    try:return codecs.lookup(enc).name
    except (LookupError,TypeError):return None
def cases():
    seen=set()
    for p in sorted(SOURCE.glob("*.po")):
        raw=p.read_bytes()
        header=re.search(br"charset=([A-Za-z0-9_.+-]+)",raw,re.IGNORECASE)
        if not header:continue
        enc=canon(header.group(1).decode("ascii"))
        if not enc:continue
        # Header declarations are masked, not used as detector hints.
        masked=re.sub(br"(?i)(charset=)[A-Za-z0-9_.+-]+",
                      lambda m:m.group(1)+b"?"*(len(m.group(0))-len(m.group(1))),raw)
        for n,off in enumerate((0,len(masked)//3,len(masked)*2//3)):
            b=masked[off:off+8192]
            # Try bounded adjustment for UTF multibyte boundary.
            for _ in range(4):
                try:
                    ref=b.decode(enc,"strict")
                    break
                except UnicodeError:
                    if n>0 and b:
                        b=b[1:]
                    if b:
                        b=b[:-1]
            else:continue
            if len(b)<128 or b.isascii():continue
            sha=hashlib.sha256(b).hexdigest()
            if sha in seen:continue
            seen.add(sha)
            yield {"name":p.name, "window":n,"sha":sha,"raw":b,"truth":enc,"expected":ref}

def score(label):
    engine=codeccat.CodecCat()
    out=[]
    for obj in cases():
        result=engine.detect(obj["raw"])
        out.append({"name":obj["name"],"window":obj["window"],"sha":obj["sha"],
                    "expected_encoding":obj["truth"],"encoding":result.encoding,
                    "correct_text":result.text==obj["expected"],
                    "reason":result.reason})
    assert len(out)>=100, f"external source insufficient: {len(out)}"
    assert len({x["name"] for x in out})>=30
    (OUT/f"glib-{label}.json").write_text(json.dumps(out,indent=2)+"\n")
    print("GLIB_EXTERNAL_SCORE",json.dumps({"label":label,"n":len(out),"correct":sum(x["correct_text"] for x in out)}))

def compare():
    orig=json.loads((OUT/"glib-baseline.json").read_text())
    new=json.loads((OUT/"glib-candidate.json").read_text())
    assert [x["sha"] for x in orig]==[x["sha"] for x in new],"input corpus diverged"
    helps=[a for a,b in zip(orig,new) if not a["correct_text"] and b["correct_text"]]
    harms=[a for a,b in zip(orig,new) if a["correct_text"] and not b["correct_text"]]
    summary={"source":"GNOME/glib", "source_commit":"d34a927488b8b664cbd305dc880e86f34f8ca467",
             "cases":len(orig),"baseline_correct":sum(x["correct_text"] for x in orig),
             "candidate_correct":sum(x["correct_text"] for x in new),
             "helps":len(helps),"harms":len(harms),
             "helped_sources":[x["name"] for x in helps],
             "harmed_sources":[x["name"] for x in harms]}
    (OUT/"glib-paired-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("GLIB_EXTERNAL_PAIRED_RESULT",json.dumps(summary),flush=True)
    assert not harms,"Candidate regressed new source"
    assert len(helps)>0,"No corroborated improvement on new source"

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--engine",choices=("baseline","candidate"))
    p.add_argument("--compare",action="store_true")
    a=p.parse_args()
    if a.compare:compare()
    else:score(a.engine)
