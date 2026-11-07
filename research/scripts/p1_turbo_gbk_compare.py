"""Research-only P1: independent HTML prescan implementations and external GBK bytes.

Never alter installed CodecCat, re-train, or upload source HTML. Archive labels are
proxies; original external-file charset claims need independent adjudication.
"""
from __future__ import annotations
import hashlib
import json
import codecs
import sys
from collections import Counter,defaultdict
from pathlib import Path
from urllib.parse import urlsplit
import zipfile

import codeccat
import turbohtml
from html5lib._inputstream import HTMLBinaryInputStream
from turbohtml.detect import detect as turbo_detect

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "native-prototype"))
from evaluate_c1_real_html import ARCHIVE, prepare, train_hashes, truth_from_label
from research_p1_html_encoding import canon, declared
from audit_holdout_overlap import blocks

REPORT=Path("research-results/p1-html-gbk-independent.json")
CANDIDATES=[
    ("CatPapers",Path("external-gbk/CatPapers/data/header.html")),
    ("CatPapers",Path("external-gbk/CatPapers/cat_papers.html")),
    ("KitJs",Path("external-gbk/KitJs/urlencode.html")),
    ("netkiller",Path("external-gbk/netkiller/architect/h5.html")),
    ("coolshell",Path("external-gbk/coolshell/articles/4261.html")),
]
ADVERSARIAL=[
    ("plain-meta",b'<html><head><meta charset="gbk"></head><body>\xd6\xd0\xce\xc4</body></html>',"gbk",None),
    ("http-equiv",b'<meta http-equiv="Content-Type" content="text/html; charset=gbk"><p>\xd6\xd0\xce\xc4</p>',"gbk",None),
    ("comment-meta",b'<!-- <meta charset="gbk"> --><p>Hello</p>',"windows-1252",None),
    ("script-fake-meta",b'<script>var html="<meta charset=gbk>";</script><p>hello</p>',"windows-1252",None),
    ("attribute-fake-meta",b'<div data-value="<meta charset=gbk>">Hi</div>',"windows-1252",None),
    ("meta-after-1024",b'X'*1024+b'<meta charset=gbk><p>\xd6\xd0</p>',"gbk",None),
    ("transport-wins",b'<meta charset=gbk><p>hello</p>',"utf-8","utf-8"),
    ("bom-wins",b'\xef\xbb\xbf<meta charset=gbk><p>hello</p>',"utf-8",None),
    ("doubled-meta",b'<meta charset=gbk><meta charset=utf-8><p>text</p>',"gbk",None),
    ("quoted-angle",b'<meta title="> embedded" charset=gbk><p>text</p>',"gbk",None),
]
def norm(label: str|None)->str|None:
    if label is None:return None
    key=canon(str(label))
    if key in ("cp932","shift_jis"):return "shift_jis"
    if key in ("gbk","gb2312","cp936"):return "gbk"
    if key:return key
    return str(label).lower().replace("-","_")

def html5_sniff(raw,transport=None):
    sniff=HTMLBinaryInputStream(raw,transport_encoding=transport,default_encoding="windows-1252",useChardet=False)
    return sniff.charEncoding[0].name

def turbo_sniff(raw,transport=None):
    parsed=turbohtml.parse(raw,encoding=transport) if transport else turbohtml.parse(raw)
    return str(parsed.encoding) if parsed.encoding else None

def has_overlap(raw,train):
    full,prefixes,spans=train
    digest=hashlib.sha256(raw).hexdigest()
    near=Counter(source for b in blocks(raw) for source in spans.get(b,()))
    return digest in full or hashlib.sha256(raw[:4096]).hexdigest() in prefixes or max(near.values(),default=0)>=3

def new_source_cases(train):
    out=[]
    for source,path in CANDIDATES:
        item={"source":source,"filename":str(path),"exists":path.is_file()}
        if not path.exists():
            out.append(item)
            continue
        raw=path.read_bytes()
        item.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                    training_overlap_candidate=has_overlap(raw,train))
        rawdecl=declared(raw)
        item["regex_meta"]=rawdecl
        try:
            item["html5lib_encoding"]=html5_sniff(raw)
            item["turbohtml_encoding"]=turbo_sniff(raw)
        except Exception as exc:
            item["parse_exception"]=type(exc).__name__
        if len(raw)<256:
            item["reason"]="too_short"
            out.append(item)
            continue
        if item["training_overlap_candidate"]:
            item["reason"]="overlap_candidate"
            out.append(item)
            continue
        try:
            gold=raw.decode("gbk","strict")
        except UnicodeError:
            item["reason"]="invalid_gbk"
            out.append(item)
            continue
        han=sum(0x3400 <= ord(ch) <= 0x9fff for ch in gold)
        item["han"]=han
        if han<10:
            item["reason"]="not_enough_han_to_claim_gbk"
            out.append(item)
            continue
        res=codeccat.detect(raw)
        item.update(reason="eligible_exploratory_not_ground_truth",
                    selected=res.encoding,candidates=list(res.candidates),
                    matches_strict_gbk=(res.text==gold),
                    recoverable_candidate=any(candidate_match(raw,c,gold) for c in res.candidates),
                    classifier_abstained=res.text is None)
        out.append(item)
    return out

def candidate_match(raw,label,text):
    try:return raw.decode(label,"strict")==text
    except (UnicodeError,LookupError):return False

def main():
    import importlib.metadata
    assert importlib.metadata.version("codeccat")=="0.1.1"
    train=train_hashes()
    ad=[]
    for label,raw,expected,transport in ADVERSARIAL:
        h=html5_sniff(raw,transport)
        t=turbo_sniff(raw,transport)
        item={"case":label,"expected_heuristic":expected,
              "html5lib":h,"turbohtml":t,
              "regex":declared(raw),
              "html5lib_matches_expected":norm(h)==norm(expected),
              "turbohtml_matches_expected":norm(t)==norm(expected)}
        ad.append(item)
    stats=defaultdict(Counter)
    samples=defaultdict(list)
    with zipfile.ZipFile(ARCHIVE) as z:
        eligible,_=prepare(z,train)
        assert len(eligible)==1886,len(eligible)
        for name,folder,codec,sha,size in eligible:
            raw=z.read(name)
            assert hashlib.sha256(raw).hexdigest()==sha
            s=stats[folder]
            s["n"]+=1
            h=html5_sniff(raw)
            t=turbo_sniff(raw)
            sr=declared(raw)
            s["html5_family_matches_directory"]+=int(norm(h)==norm(codec))
            s["turbohtml_family_matches_directory"]+=int(norm(t)==norm(codec))
            s["html5_and_turbohtml_agree"]+=int(norm(h)==norm(t))
            s["html5_and_turbohtml_disagree"]+=int(norm(h)!=norm(t))
            s["turbohtml_vs_regex_differ"]+=int(norm(t)!=norm(sr)) if sr else 0
            if norm(h)!=norm(t) and len(samples[folder])<10:
                samples[folder].append({"sha256":sha,"bytes":size,"html5lib":h,
                                        "turbohtml":t,"regex":sr})
    newcases=new_source_cases(train)
    out={
        "scope":"P1 Research only; unchanged published CodecCat 0.1.1",
        "new_external_gbk":newcases,
        "adversarial":ad,
        "archive_stats":{k:dict(v) for k,v in sorted(stats.items())},
        "archive_disagreement_metadata_only":dict(samples),
        "limitations":[
            "Document labels are not adjudicated ground truth.",
            "html5lib and turbohtml are external implementations and conformance must be tested, not assumed.",
            "No archived HTTP Content-Type available.",
            "External GBK files may themselves be misdeclared or have related provenance.",
            "Adversarial fixture expected values are only those declared in handcrafted setup.",
        ],
        "raw_bytes_exported":False,
    }
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(out,indent=2,ensure_ascii=True,sort_keys=True)+"\n")
    print("P1_TURBO_ADVERSARIAL",json.dumps(ad,sort_keys=True))
    print("P1_TURBO_ARCHIVE",json.dumps(out["archive_stats"],sort_keys=True))
    print("P1_GBK_NEW_SOURCES",json.dumps(newcases,sort_keys=True))

if __name__=="__main__":
    main()
