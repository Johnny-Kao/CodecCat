"""Single-use confirmation on preregistered Lynx/gettext native PO bytes.

Corpus was locked in Actions #41240397820 before this script was created.
None of the external PO bytes are used for training. Metric is STRICT
decoded Unicode equality, not exact historical encoding label. The windows
within a source catalog are correlated and never treated as independent
documents for uncertainty claims.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import codecs
import json
from pathlib import Path
import subprocess
import sys

import chardet
import charset_normalizer
from charset_normalizer import from_bytes

from codeccat.vnext_decoding import decode
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_selective import SelectiveLatinDetector
from codeccat.vnext_utf8_evidence import Utf8EvidenceDetector

sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_legacy_po_freeze import SOURCES, sha
from vnext_vim_catalog_freeze import catalog_samples, fragments, take_window, POSITIONS, WINDOW
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_content_safe_contrast import contrastive_augmentation
from vnext_training_strategy_audit import strict_text

FROZEN_FINGERPRINT="5b26415a5fbe016dd25789a1a55508d7c5ad84f1912cf753b40002c162ccfe3e"
TRAIN_REV="b0c0d206913ba118ffea2f7d306883e7a4a76db7"
METHODS=("baseline","latin","selective","utf8","combined","chardet7","charset_normalizer")


def pin(root,expected):
    actual=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if actual!=expected:
        raise RuntimeError(f"unapproved source revision: {actual}")


def frozen_rows(manifest_path,source_roots):
    manifest=json.loads(Path(manifest_path).read_text())
    if (manifest.get("fingerprint")!=FROZEN_FINGERPRINT or
            manifest.get("schema")!="codeccat-legacy-po-external-frozen-v1"):
        raise RuntimeError("source manifest not preregistered")
    for name,revision,paths in SOURCES:
        pin(source_roots[name],revision)
    rows=[]
    for section in manifest["recorded"]:
        source=section["source"]
        original=section["catalog"]
        f=(source_roots[source]/original["path"]).resolve()
        if not f.is_relative_to(source_roots[source].resolve()):
            raise RuntimeError("source file escapes root")
        now, ledger=catalog_samples(f)
        # Lock filename metadata and original bytes, not just the labels.
        ledger["path"]=f.relative_to(source_roots[source]).as_posix()
        for item in now:
            item["path"]=ledger["path"]
        if ledger!=original or now!=section["windows"]:
            raise RuntimeError("raw file or byte-window selection changed after freeze")
        payload=fragments(f.read_bytes())
        for window in now:
            i=window["window"]
            start=round((len(payload)-WINDOW)*POSITIONS[i])
            raw=take_window(payload,max(0,start),window["codec"])
            if raw is None or sha(raw)!=window["sha256"]:
                raise RuntimeError("byte window changed")
            text=raw.decode(window["codec"],"strict")
            assert sha(text.encode("utf-8","surrogatepass"))==window["decoded_sha256"]
            rows.append({
                "source":source,"file":window["path"],"window":i,
                "data":raw,"text":text,"codec":window["codec"],
                "sha256":window["sha256"],
            })
    if len(rows)!=manifest["n_utf8_windows"]+manifest["n_legacy_windows"]:
        raise RuntimeError("frozen window count changed")
    return rows


def competitor(name,raw):
    if name=="chardet7":
        encoding=chardet.detect(raw).get("encoding")
    else:
        best=from_bytes(raw).best()
        encoding=best.encoding if best is not None else None
    try:
        text=raw.decode(encoding,"strict") if encoding else None
    except (LookupError,UnicodeError,ValueError,TypeError):
        text=None
    return encoding,text


def quality(rows):
    outcomes={"n":len(rows),"overall":{},"paired":{},"by_codec":{},"by_file":{},"by_source":{}}
    for name in METHODS:
        sub=[r["predictions"][name] for r in rows]
        outcomes["overall"][name]={
            "n":len(sub),
            "text_correct":sum(p["text_correct"] for p in sub),
            "label_correct":sum(p["label_correct"] for p in sub),
            "abstained":sum(p["abstained"] for p in sub),
        }
    for name in METHODS[1:]:
        helps=sum(r["predictions"][name]["text_correct"] and not r["predictions"]["baseline"]["text_correct"] for r in rows)
        harms=sum(not r["predictions"][name]["text_correct"] and r["predictions"]["baseline"]["text_correct"] for r in rows)
        outcomes["paired"][name]={"helps":helps,"harms":harms,"net":helps-harms}
    for dimension,field in (("by_codec","codec"),("by_file","file"),("by_source","source")):
        for k in sorted(set(r[field] for r in rows)):
            sub=[r for r in rows if r[field]==k]
            outcomes[dimension][k]={"n":len(sub),**{
                name:sum(r["predictions"][name]["text_correct"] for r in sub)
                for name in METHODS
            }}
    return outcomes


def main(args):
    sources={"lynx":Path(args.lynx).resolve(),"gettext":Path(args.gettext).resolve()}
    pin(args.training,TRAIN_REV)
    if chardet.__version__!="7.7.0" or charset_normalizer.__version__!="3.5.2":
        raise RuntimeError("unpinned third party versions")
    test=frozen_rows(args.manifest,sources)
    training=[(raw,label,group) for raw,label,group,fold,digest in corpus(Path(args.training))]
    training_texts={sha(strict_text(b,y).encode("utf-8","surrogatepass")) for b,y,g in training}
    seen=set()
    clean=[]
    excluded=Counter()
    for row in test:
        tid=sha(row["text"].encode("utf-8","surrogatepass"))
        if tid in training_texts:
            excluded["training_text_overlap"]+=1
            continue
        if row["sha256"] in seen:
            excluded["duplicate_test_bytes"]+=1
            continue
        seen.add(row["sha256"])
        clean.append(row)
    by_family=Counter("utf8" if r["codec"]=="utf-8" else "legacy" for r in clean)
    if len(clean)<65 or by_family["legacy"]<35 or by_family["utf8"]<20:
        raise RuntimeError(f"not enough uncontaminated cross-family validation: {len(clean)}, {dict(by_family)}")

    residual,_=residual_subset(training)
    contrast,info=contrastive_augmentation(residual,per_direction=24,exclude_ebcdic=True)
    baseline=TrainableDetector.fit(residual,feature_profile="stable23")
    latin=TrainableDetector.fit(residual+contrast,feature_profile="stable23")
    selective=SelectiveLatinDetector(baseline,latin)
    models={
        "baseline":baseline,"latin":latin,"selective":selective,
        "utf8":Utf8EvidenceDetector(baseline,min_codepoints=16),
        "combined":Utf8EvidenceDetector(selective,min_codepoints=16),
    }
    comparisons=[]
    for row in clean:
        raw=row["data"];expected=row["text"];true_codec=row["codec"]
        results={}
        for name,model in models.items():
            outcome=decode(model,raw,bom_policy="preserve")
            try:
                exact=bool(outcome.encoding and codecs.lookup(outcome.encoding).name==true_codec)
            except LookupError:
                exact=False
            results[name]={
                "codec":outcome.encoding,"text_correct":outcome.text==expected,
                "label_correct":exact,"ambiguous":outcome.ambiguous,
                "abstained":outcome.text is None,
            }
        for name in ("chardet7","charset_normalizer"):
            label,text=competitor(name,raw)
            try:
                exact=bool(label and codecs.lookup(label).name==true_codec)
            except LookupError:
                exact=False
            results[name]={
                "codec":label,"text_correct":text==expected,
                "label_correct":exact,"abstained":text is None,
            }
        comparisons.append({
            "source":row["source"],"file":row["file"],"window":row["window"],
            "sha256":row["sha256"],"codec":true_codec,"predictions":results,
        })
    score=quality(comparisons)
    # This is a guardrail, not acceptance: a zero-error single suite is
    # still insufficient to certify broad population safety.
    report={
        "schema":"codeccat-frozen-legacy-po-external-one-shot-v1",
        "status":"CONSUMED_EXTERNAL_EVIDENCE; NOT_PRODUCTION_APPROVAL",
        "source_freeze_run":"https://github.com/Johnny-Kao/CodecCat/actions/runs/41240397820",
        "frozen_source_sha":FROZEN_FINGERPRINT,
        "source_revisions":{name:sha for name,sha,_ in SOURCES},
        "training_revision":TRAIN_REV,
        "n_frozen":len(test),"n_tested":len(clean),
        "excluded":dict(excluded),"n_tested_by_family":dict(by_family),
        "metric":"strict Unicode text equality; label correctness independently reported; no labels supplied to decoder",
        "selection":"all non-UTF8 files plus 6 preselected UTF8 control files per source",
        "outcomes":score,
        "warnings":[
            "The baseline was trained on chardet/test-data; exact content overlap excluded, near duplicates not ruled out",
            "Multiple non-independent byte windows from each translation catalog",
            "No external native source payloads committed or used for model fitting",
            "Source release files can be converted variants; historical native authoring unverified",
            "After evaluation the independent corpus is consumed and cannot be a fresh holdout",
        ],
    }
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"cases.jsonl").write_text("\n".join(json.dumps(row,sort_keys=True) for row in comparisons)+"\n")
    print("LEGACY_EXTERNAL_ONE_SHOT_BEGIN",flush=True)
    print(json.dumps({k:report[k] for k in ("n_tested","excluded","n_tested_by_family","outcomes")},indent=2,sort_keys=True),flush=True)
    print("LEGACY_EXTERNAL_ONE_SHOT_END",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--lynx",required=True)
    p.add_argument("--gettext",required=True)
    p.add_argument("--training",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",default="research-results/legacy-po-one-shot")
    main(p.parse_args())
