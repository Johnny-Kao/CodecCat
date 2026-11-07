"""P1 research 1/2/3: old benchmark semantics, HTML-aware prescan, independent GBK probes.
Research-only: no Runtime edits, no default model, no source bytes in uploaded reports.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import importlib.metadata

import codeccat
from html5lib._inputstream import HTMLBinaryInputStream

sys.path.insert(0,str(Path("native-prototype").resolve()))
from evaluate_c1_real_html import ARCHIVE,prepare,train_hashes,truth_from_label
from audit_holdout_overlap import blocks
from research_p1_html_encoding import declared,canon,source_mask

OUTPUT=Path("research-results/p1-three-track-audit.json")
GBK_FILES=[
    ("piconv",Path("independent-piconv/data-set/chinese-ansi.txt")),
    ("piconv",Path("independent-piconv/test/iconv-test/gbk-gbk.c")),
    ("piconv",Path("independent-piconv/test/iconv-test-rec/gbk-gbk.c")),
    ("cpython_previously_examined",Path("independent-cpython/Lib/test/cjkencodings/gbk.txt")),
]
NEGATIVE=[
    ("charset_normalizer_utf8_control",Path("independent-charset-normalizer/data/sample-chinese.txt")),
]
ADVERSARIAL=[
    ("meta_normal",b'<meta charset="gbk"><p>\xd6\xd0\xce\xc4</p>',"gbk"),
    ("meta_http_equiv",b'<meta http-equiv="Content-Type" content="text/html; charset=gbk"><p>\xd6\xd0\xce\xc4</p>',"gbk"),
    ("meta_inside_comment",b'<!-- <meta charset=gbk> --> <p>hello</p>',"windows-1252"),
    ("meta_inside_script",b'<script>var x="<meta charset=gbk>";</script><p>hello</p>',"windows-1252"),
    ("meta_beyond_1024",b"x"*1024+b"<meta charset=gbk>","windows-1252"),
    ("transport_beats_meta",b"<meta charset=gbk><p>hi</p>","utf-8"),
    ("bom_beats_meta",b"\xef\xbb\xbf<meta charset=gbk><p>hi</p>","utf-8"),
]
def sniff(raw:bytes,transport:str|None=None):
    obj=HTMLBinaryInputStream(raw,transport_encoding=transport,
                              default_encoding="windows-1252",useChardet=False)
    return obj.charEncoding[0].name,obj.charEncoding[1]

def norm(name:str)->str:
    return canon(name) or name.lower().replace("_","-")

def audit_overlap(raw:bytes, historic):
    full,prefixes,spans=historic
    digest=hashlib.sha256(raw).hexdigest()
    near=Counter(p for b in blocks(raw) for p in spans.get(b,()))
    return digest not in full and hashlib.sha256(raw[:4096]).hexdigest() not in prefixes and max(near.values(),default=0)<3

def independent_cases(historic):
    out=[]
    for category,p in GBK_FILES+NEGATIVE:
        if not p.is_file():
            out.append({"source":category,"path":str(p),"status":"missing"})
            continue
        raw=p.read_bytes()
        label="gbk" if category!="charset_normalizer_utf8_control" else "utf-8"
        item={"source":category,"path":str(p),"bytes":len(raw),
              "sha256":hashlib.sha256(raw).hexdigest(),"label":label,
              "historical_training_screen_negative":audit_overlap(raw,historic)}
        try:
            expected=raw.decode(label,"strict")
        except UnicodeError:
            item["status"]="label_invalid";out.append(item);continue
        pred=codeccat.detect(raw)
        item.update(status="evaluated",selected=pred.encoding,
                    candidates=list(pred.candidates),
                    unicode_agrees_with_declared_label=(pred.text==expected),
                    any_candidate_exact=any(
                        decode_match(raw,x,expected) for x in pred.candidates
                    ),abstained=pred.text is None,
                    format="original_source_bytes_not_recoded")
        out.append(item)
    return out

def decode_match(raw,label,expected):
    try:
        return raw.decode(label,"strict")==expected
    except (UnicodeError,LookupError):
        return False

def main():
    assert importlib.metadata.version("codeccat")=="0.1.1"
    historic=train_hashes()
    adversarial=[]
    for name,raw,expected in ADVERSARIAL:
        # transport only applies to explicit labeled case
        transport="utf-8" if name=="transport_beats_meta" else None
        actual,confidence=sniff(raw,transport=transport)
        adversarial.append({"name":name,"expected":expected,
                            "selected":actual,"confidence":confidence,
                            "matches_expected":norm(actual)==norm(expected),
                            "regex_guess":declared(raw)})
    groups=defaultdict(Counter)
    mismatches=defaultdict(list)
    detector=codeccat.CodecCat()
    with zipfile.ZipFile(ARCHIVE) as z:
        eligible,_=prepare(z,historic)
        assert len(eligible)==1886,len(eligible)
        for name,folder,label,sha,sz in eligible:
            raw=z.read(name)
            assert hashlib.sha256(raw).hexdigest()==sha
            s=groups[folder]
            s["n"]+=1
            detected=detector.detect(raw)
            gold=truth_from_label(raw,label)
            baseline=(detected.text==gold)
            s["base_label_agree"]+=int(baseline)
            parse,confidence=sniff(raw)
            prescan=declared(raw)
            s["sniff_matches_archive_family"]+=int(norm(parse)==norm(label))
            s["sniff_differs_archive_family"]+=int(norm(parse)!=norm(label))
            s["sniff_differs_regex"]+=int(norm(parse)!=norm(prescan) if prescan else True)
            s["sniff_certain"]+=int(confidence=="certain")
            s["sniff_tentative"]+=int(confidence=="tentative")
            if norm(parse)==norm(label):
                s["html_context_agree_when_matched"]+=1
                s["html_context_could_help_baseline_disagree"]+=int(not baseline)
            elif len(mismatches[folder])<12:
                mismatches[folder].append({
                    "sha256":sha,"source_name_hash":hashlib.sha256(name.encode()).hexdigest(),
                    "folder_label":label,"sniffed":parse,"confidence":confidence,
                    "regex_meta":prescan,"baseline_guess":detected.encoding,
                })
    independent=independent_cases(historic)
    report={
        "schema":"codeccat-p1-three-track-audit-v1",
        "runtime":"codeccat==0.1.1",
        "track_1":{"description":"frozen source-vs-oracle benchmarks unchanged; use separate Python vs WHATWG scoring established in PR54",
                   "prior_dual_oracle":{"Shift_JIS":{"python":239,"whatwg":516,"n":532},
                                         "GBK":{"python":115,"whatwg":115,"n":356}}},
        "track_2":{"description":"HTMLBinaryInputStream approximate WHATWG-compatible encoding sniff with no HTTP transport on crawled files; compare lexical regex and directory labels",
                   "groups":{k:dict(v) for k,v in sorted(groups.items())},
                   "adversarial":adversarial,"nonmatching_metadata":dict(mismatches),
                   "limitations":"html5lib implementation isn't a browser; no real HTTP headers; archive labels not independently verified; no lossless correctness claim"},
        "track_3":{"description":"new piconv source bytes + previously examined CPython fixture + charset-normalizer negative control",
                   "cases":independent,
                   "limitations":"very small heterogeneous samples; labels from own projects; CPython previously examined; overlap screen is finite; insufficient to infer new-model production accuracy"},
        "raw_bytes_exported":False,
        "release_authorized":False,
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("P1_TRACK2_SUMMARY",json.dumps(report["track_2"]["groups"],sort_keys=True))
    print("P1_TRACK2_ADVERSARIAL",json.dumps(adversarial,sort_keys=True))
    print("P1_TRACK3_INDEPENDENT",json.dumps(independent,sort_keys=True))

if __name__=="__main__":
    main()
