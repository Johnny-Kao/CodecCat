"""Single-use unseen Gawk/Bash native-byte generalization test of frozen L2/L3.

Source byte windows preregistered in GitHub Actions run 41241122649
before this scoring script was written. No parameters are chosen from
these results; the current L3 and UTF8 settings are unchanged.
"""
from __future__ import annotations
import argparse,codecs,collections,json
from pathlib import Path
import subprocess,sys
import chardet,charset_normalizer
from charset_normalizer import from_bytes

from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_selective import SelectiveLatinDetector
from codeccat.vnext_utf8_evidence import Utf8EvidenceDetector
from codeccat.vnext_l3_language import LanguageCoherence
from codeccat.vnext_decoding import decode

sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_new_gnu_source_freeze import SOURCES,SCHEMA
from vnext_legacy_po_freeze import sha
from vnext_vim_catalog_freeze import catalog_samples,fragments,take_window,POSITIONS,WINDOW
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_content_safe_contrast import contrastive_augmentation
from vnext_training_strategy_audit import strict_text

FROZEN="7e11e0d76121103b6e30ba23f3f64e0416f15b65214efc76fc7b50edf098300e"
TRAIN_REV="b0c0d206913ba118ffea2f7d306883e7a4a76db7"
MODES=("L2","L2_L3","selective_L3","utf8_L3","chardet7","charset_normalizer")


def pin(root,sha_expected):
    actual=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if actual!=sha_expected:raise RuntimeError(f"unexpected git revision: {actual}")


def read_frozen(roots,manifest):
    j=json.loads(Path(manifest).read_text())
    if j.get("source_digest")!=FROZEN or j.get("schema")!=SCHEMA:
        raise RuntimeError("unfrozen source manifest")
    for name,(revision,_) in SOURCES.items():
        pin(roots[name],revision)
    test=[]
    for record in j["records"]:
        label=record["source"]
        catalog=record["catalog"]
        f=(roots[label]/catalog["path"]).resolve()
        if not f.is_relative_to(roots[label].resolve()):
            raise RuntimeError("path traversal")
        samples,ledger=catalog_samples(f)
        ledger["path"]=f.relative_to(roots[label]).as_posix()
        for row in samples:
            row["path"]=ledger["path"]
        if ledger!=catalog or samples!=record["windows"]:
            raise RuntimeError(f"source {label}/{catalog['path']} changed after preregistration")
        data=fragments(f.read_bytes())
        for row in samples:
            position=round((len(data)-WINDOW)*POSITIONS[row["window"]])
            raw=take_window(data,max(0,position),row["codec"])
            if raw is None or sha(raw)!=row["sha256"]:
                raise RuntimeError("actual raw bytes differ from preregistration")
            text=raw.decode(row["codec"],"strict")
            if sha(text.encode("utf-8","surrogatepass"))!=row["decoded_sha256"]:
                raise RuntimeError("labeled decode changed after preregistration")
            test.append({"source":label,"file":catalog["path"],"codec":row["codec"],
                         "raw":raw,"text":text,"sha256":row["sha256"],
                         "window":row["window"]})
    return test


def cmp_other(which,raw):
    if which=="chardet7":
        label=chardet.detect(raw).get("encoding")
    else:
        result=from_bytes(raw).best()
        label=result.encoding if result is not None else None
    try:
        text=raw.decode(label,"strict") if label else None
    except (UnicodeError,ValueError,LookupError,TypeError):
        text=None
    return label,text


def aggregate(rows):
    metrics={"n":len(rows),"overall":{},"paired_vs_L2":{},"by_codec":{},
             "by_source":{},"by_catalog":{}}
    for key in MODES:
        col=[r["predictions"][key] for r in rows]
        metrics["overall"][key]={
            "correct_unicode":sum(c["text_correct"] for c in col),
            "correct_label":sum(c["label_correct"] for c in col),
            "abstain":sum(c["abstained"] for c in col),
            "l3_called":sum(c.get("l3_called",False) for c in col),
            "l3_changed":sum(c.get("l3_changed",False) for c in col),
        }
    for mode in MODES[1:]:
        helps=sum(r["predictions"][mode]["text_correct"] and
                  not r["predictions"]["L2"]["text_correct"] for r in rows)
        harms=sum(r["predictions"]["L2"]["text_correct"] and
                  not r["predictions"][mode]["text_correct"] for r in rows)
        metrics["paired_vs_L2"][mode]={"helps":helps,"harms":harms,"net":helps-harms}
    for dim,field in (("by_codec","codec"),("by_source","source"),("by_catalog","file")):
        for name in sorted({r[field] for r in rows}):
            subset=[r for r in rows if r[field]==name]
            metrics[dim][name]={"n":len(subset),**{
                mode:sum(r["predictions"][mode]["text_correct"] for r in subset)
                for mode in MODES
            }}
    return metrics


def run(args):
    roots={"gawk":Path(args.gawk).resolve(),"bash":Path(args.bash).resolve()}
    pin(args.training,TRAIN_REV)
    if chardet.__version__!="7.7.0" or charset_normalizer.__version__!="3.5.2":
        raise RuntimeError("unpinned comparator")
    frozen=read_frozen(roots,args.manifest)
    training=[(b,y,g) for b,y,g,fold,digest in corpus(Path(args.training))]
    seen_text={sha(strict_text(b,y).encode("utf-8","surrogatepass")) for b,y,g in training}
    held=[];seen_raw=set();removed=collections.Counter()
    for item in frozen:
        tid=sha(item["text"].encode("utf-8","surrogatepass"))
        if tid in seen_text:
            removed["training_text_overlap"]+=1
        elif item["sha256"] in seen_raw:
            removed["duplicate_raw_window"]+=1
        else:
            seen_raw.add(item["sha256"]);held.append(item)
    codec_counts=collections.Counter("utf8" if r["codec"]=="utf-8" else "legacy" for r in held)
    if len(held)<60 or codec_counts["legacy"]<20 or codec_counts["utf8"]<20:
        raise RuntimeError(f"too few decontaminated original bytes: {len(held)}, {codec_counts}")

    residual,_=residual_subset(training)
    augment,_=contrastive_augmentation(residual,per_direction=24,exclude_ebcdic=True)
    baseline=TrainableDetector.fit(residual,feature_profile="stable23")
    latin=TrainableDetector.fit(residual+augment,feature_profile="stable23")
    language=LanguageCoherence.fit(training)
    models={
        "L2":(baseline,None),
        "L2_L3":(baseline,language),
        "selective_L3":(SelectiveLatinDetector(baseline,latin),language),
        "utf8_L3":(Utf8EvidenceDetector(baseline,min_codepoints=16),language),
    }
    records=[]
    for row in held:
        preds={}
        for name,(model,l3) in models.items():
            result=decode(model,row["raw"],bom_policy="preserve",language_model=l3)
            try:match=bool(result.encoding and
                           codecs.lookup(result.encoding).name==row["codec"])
            except LookupError:match=False
            preds[name]={"encoding":result.encoding,"text_correct":result.text==row["text"],
                         "label_correct":match,"abstained":result.text is None,
                         "l3_called":result.l3_called,"l3_changed":result.l3_changed}
        for name in ("chardet7","charset_normalizer"):
            detected,text=cmp_other(name,row["raw"])
            try:match=bool(detected and codecs.lookup(detected).name==row["codec"])
            except LookupError:match=False
            preds[name]={"encoding":detected,"text_correct":text==row["text"],
                         "label_correct":match,"abstained":text is None}
        records.append({"file":row["file"],"source":row["source"],"window":row["window"],
                        "sha256":row["sha256"],"codec":row["codec"],"predictions":preds})
    summary=aggregate(records)
    report={
        "schema":"codeccat-fresh-gawk-bash-l3-confirmation-v1",
        "status":"CONSUMED_EXTERNAL_SOURCE; NO_AUTOMATIC_PRODUCTION_APPROVAL",
        "frozen_source_run":"https://github.com/Johnny-Kao/CodecCat/actions/runs/41241122649",
        "frozen_selection_digest":FROZEN,
        "training_sha":TRAIN_REV,
        "n_frozen":len(frozen),"n_tested":len(held),
        "source_classes":dict(codec_counts),"removed":dict(removed),
        "model":"existing immutable stable23 and L3 thresholds; no tuning for this source",
        "results":summary,
        "limitations":[
            "Multiple windows per original file are correlated",
            "Inclusion and byte labels were preregistered before any model output",
            "This is not a representative prevalence sample",
            "New source has now been used and is no longer available as future blind holdout",
        ],
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"cases.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True) for x in records)+"\n")
    print("GNU_UNSEEN_L3_CONFIRMATION_BEGIN",flush=True)
    print(json.dumps({k:report[k] for k in ("n_frozen","n_tested","source_classes","removed","results")},indent=2,sort_keys=True),flush=True)
    print("GNU_UNSEEN_L3_CONFIRMATION_END",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gawk",required=True)
    p.add_argument("--bash",required=True)
    p.add_argument("--training",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",default="research-results/gawk-bash-one-shot")
    run(p.parse_args())
