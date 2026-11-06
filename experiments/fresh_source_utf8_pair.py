"""Pre-pinned new source: GNOME/gnome-software translations.

Compares installed published a4 vs staged candidate with same model bytes.
Input source fixed before scoring. Exact text is the primary outcome.
"""
import argparse,codecs,hashlib,json,re
from pathlib import Path
from codeccat import CodecCat

ROOT=Path("fresh-source/po")
OUT=Path("research-results/gnome-software-utf8")
COMMIT="d693edda74a6ae3a5360a6d19ab558c2fd3ac8cb"

def canon(s):
    try:return codecs.lookup(s).name
    except (TypeError,LookupError):return None

def samples():
    result=[]; seen=set()
    for path in sorted(ROOT.glob("*.po")):
        raw=path.read_bytes()
        m=re.search(br"charset=([A-Za-z0-9_.+-]+)",raw,re.I)
        if not m:continue
        codec=canon(m.group(1).decode("ascii"))
        if not codec:continue
        raw=re.sub(br"(?i)(charset=)[A-Za-z0-9_.+-]+",
                   lambda m:m.group(1)+b"?"*(len(m.group(0))-len(m.group(1))),raw)
        for index,offset in enumerate((0,len(raw)//3,2*len(raw)//3)):
            block=raw[offset:offset+8192]
            for _ in range(8):
                try:text=block.decode(codec,"strict").lstrip("\ufeff");break
                except UnicodeError:
                    if index>0 and block:block=block[1:]
                    if block:block=block[:-1]
            else:continue
            if len(block)<128 or block.isascii():continue
            digest=hashlib.sha256(block).hexdigest()
            if digest in seen:continue
            seen.add(digest)
            result.append({"file":path.name,"window":index,"raw":block,
                           "truth":codec,"text":text,"sha256":digest})
    assert len(result)>=150,(len(result),"not enough cases")
    return result

def score(tag):
    model=CodecCat()
    data=[]
    for r in samples():
        pred=model.detect(r["raw"])
        data.append({"file":r["file"],"window":r["window"],"sha256":r["sha256"],
                     "truth":r["truth"],"pred":pred.encoding,"correct_text":pred.text==r["text"]})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f"{tag}.json").write_text(json.dumps(data,indent=2)+"\n")
    print("FRESH_SOURCE_SCORE",json.dumps({"tag":tag,"n":len(data),
        "correct":sum(r["correct_text"] for r in data)}),flush=True)

def compare():
    old=json.loads((OUT/"a4.json").read_text())
    new=json.loads((OUT/"candidate.json").read_text())
    assert [(r["file"],r["sha256"]) for r in old]==[(r["file"],r["sha256"]) for r in new]
    helpers=[r for r,s in zip(old,new) if not r["correct_text"] and s["correct_text"]]
    harmed=[r for r,s in zip(old,new) if r["correct_text"] and not s["correct_text"]]
    result={"repository":"GNOME/gnome-software","commit":COMMIT,"n":len(old),
            "old_correct":sum(r["correct_text"] for r in old),
            "new_correct":sum(r["correct_text"] for r in new),
            "helped":len(helpers),"harmed":len(harmed),
            "harmed_sources":[r["file"] for r in harmed]}
    (OUT/"paired-summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("FRESH_SOURCE_PAIRED",json.dumps(result),flush=True)
    assert not harmed, "new source has decoding regressions"
    assert helpers, "no observed improvement on new source"
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--tag",choices=("a4","candidate"))
    p.add_argument("--compare",action="store_true")
    a=p.parse_args()
    if a.compare:compare()
    else:score(a.tag)
