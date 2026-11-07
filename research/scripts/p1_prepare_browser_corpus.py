"""Select parser-disagreement original HTML fixtures for actual browser arbitration.

Private ephemeral CI fixtures: NEVER upload generated raw HTML or manifest (URLs
and HTML contents); only JSON aggregates from the browser run are publishable.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json
import sys
import zipfile

from html5lib._inputstream import HTMLBinaryInputStream
import turbohtml
sys.path.insert(0,str(Path("native-prototype").resolve()))
from evaluate_c1_real_html import ARCHIVE,prepare,train_hashes
from research_p1_html_encoding import canon

DIR=Path("private-browser-fixtures")
MANIFEST=DIR/"manifest.json"
def norm(name):
    if name is None:return None
    try:
        n=canon(str(name))
    except Exception:
        n=None
    return n or str(name).lower().replace("-","_")
def main():
    historic=train_hashes()
    DIR.mkdir(exist_ok=True)
    selected=[]
    same=defaultdict(int)
    differing=Counter()
    with zipfile.ZipFile(ARCHIVE) as z:
        eligible,_=prepare(z,historic)
        assert len(eligible)==1886
        for path,folder,codec,sha,size in eligible:
            raw=z.read(path)
            h=HTMLBinaryInputStream(raw,default_encoding="windows-1252",useChardet=False).charEncoding[0].name
            t=str(turbohtml.parse(raw).encoding)
            mismatch=norm(h)!=norm(t)
            if mismatch:
                differing[folder]+=1
            if not mismatch:
                if same[folder]>=12:continue
                same[folder]+=1
            id=len(selected)
            file=DIR/f"{id:04d}.html"
            file.write_bytes(raw)
            selected.append({"id":id,"sha256":sha,"folder":folder,"label":codec,
                             "html5lib":h,"turbohtml":t,
                             "parsers_disagree":mismatch,"bytes":size,
                             "file":file.name})
    MANIFEST.write_text(json.dumps(selected,indent=2)+"\n",encoding="utf-8")
    print("P1_BROWSER_PRIVATE_SET",json.dumps({"total":len(selected),
          "disagreements":dict(differing),"agreement_controls":dict(same)},
           sort_keys=True))
if __name__=="__main__":
    main()
