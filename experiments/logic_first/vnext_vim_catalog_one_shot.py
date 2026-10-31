"""One-shot accuracy check on authentic Vim translation catalog bytes.

Source, window scheme and SHA frozen BEFORE any model scoring in
https://github.com/Johnny-Kao/CodecCat/actions/runs/41238184283

Only actual bytes extracted from source-controlled PO msgstr records are used.
Charset declarations are excluded; the original source data is not
redistributed or used for fitting. Source catalog windows are correlated,
and some are build-generated from UTF-8 originals.
"""
from __future__ import annotations
import argparse,codecs,collections,hashlib,json,pathlib,platform,subprocess,sys
import chardet,charset_normalizer
from charset_normalizer import from_bytes
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_selective import SelectiveLatinDetector
from codeccat.vnext_decoding import decode

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_content_safe_contrast import contrastive_augmentation
from vnext_training_strategy_audit import strict_text
from vnext_vim_catalog_freeze import catalog_samples,sha,VIM_REV

FINGERPRINT="3e20642290ef06a1f1dc7ccd19c3a1264601471aea5989965beb7c96d063c992"
TRAIN_REV="b0c0d206913ba118ffea2f7d306883e7a4a76db7"
MODELS=("stable23","latin_specialist","selective_router","chardet7","charset_normalizer")


def verify_commit(folder,expected):
    actual=subprocess.check_output(["git","-C",str(folder),"rev-parse","HEAD"],text=True).strip()
    if actual!=expected: raise RuntimeError(f"repository pinned SHA differs: {actual}")


def get_frozen(root,manifest):
    freeze=json.loads(pathlib.Path(manifest).read_text())
    if freeze.get("fingerprint")!=FINGERPRINT or freeze.get("source_revision")!=VIM_REV:
        raise RuntimeError("source manifest must be frozen before evaluation")
    expected=freeze["windows"]
    generated=[]
    for filename in sorted((pathlib.Path(root)/"src"/"po").glob("*.po")):
        samples,rec=catalog_samples(filename)
        generated.extend(samples)
        catalog=next((x for x in freeze["catalogs"] if x["path"]==filename.name),None)
        if not catalog or catalog.get("file_sha256")!=rec.get("file_sha256"):
            raise RuntimeError(f"original catalog changed: {filename.name}")
    if generated!=expected: raise RuntimeError("window selection no longer matches frozen manifest")
    # Recompute in the same function to ensure raw substrings are the exact
    # source controlled bytes, never synthesizing translated text.
    raw_map={}
    for filename in sorted((pathlib.Path(root)/"src"/"po").glob("*.po")):
        raw=filename.read_bytes()
        from vnext_vim_catalog_freeze import HEADER,fragments,take_window,POSITIONS,WINDOW
        match=HEADER.search(raw[:15000])
        if match is None:continue
        try: codec=codecs.lookup(match.group(1).decode("ascii")).name
        except LookupError:continue
        payload=fragments(raw)
        if len(payload)<1024:continue
        for idx,rel in enumerate(POSITIONS):
            offset=round((len(payload)-WINDOW)*rel)
            rawpiece=take_window(payload,max(0,offset),codec)
            if rawpiece is not None:
                raw_map[(filename.name,idx)]=rawpiece
    records=[]
    for row in expected:
        content=raw_map.get((row["file"],row["window"]))
        if content is None or sha(content)!=row["sha256"]:
            raise RuntimeError("actual source bytes mismatch manifest")
        text=content.decode(row["codec"],"strict")
        if sha(text.encode("utf-8","surrogatepass"))!=row["decoded_sha256"]:
            raise RuntimeError("strict source decode mismatch")
        records.append({
            "raw":content,"text":text,"file":row["file"],
            "codec":row["codec"],"sha":row["sha256"],"window":row["window"],
            "generated":next(x["generated_by_build"] for x in freeze["catalogs"] if x["path"]==row["file"]),
        })
    return records


def third_party(name,raw):
    try:
        if name=="chardet7":
            label=chardet.detect(raw).get("encoding")
        else:
            match=from_bytes(raw).best()
            label=match.encoding if match is not None else None
        text=raw.decode(label,"strict") if label else None
    except (UnicodeError,LookupError,ValueError,TypeError):
        label=None;text=None
    return label,text


def stats(cases):
    result={"n":len(cases)}
    for name in MODELS:
        rows=[r["predictions"][name] for r in cases]
        result[name]={
            "correct_unicode":sum(r["text_correct"] for r in rows),
            "correct_label":sum(r["label_correct"] for r in rows),
            "abstained":sum(r["abstained"] for r in rows),
            "n":len(rows),
        }
    result["paired"]={}
    for name in MODELS[1:]:
        h=sum(r["predictions"][name]["text_correct"] and
              not r["predictions"]["stable23"]["text_correct"] for r in cases)
        d=sum(r["predictions"]["stable23"]["text_correct"] and
              not r["predictions"][name]["text_correct"] for r in cases)
        result["paired"][name]={"helped":h,"harmed":d,"net":h-d}
    per_catalog={}
    for f in sorted({r["file"] for r in cases}):
        group=[r for r in cases if r["file"]==f]
        per_catalog[f]={"n":len(group),
                        **{name:sum(r["predictions"][name]["text_correct"] for r in group)
                           for name in MODELS}}
    per_codec={}
    for f in sorted({r["codec"] for r in cases}):
        group=[r for r in cases if r["codec"]==f]
        per_codec[f]={"n":len(group),
                      **{name:sum(r["predictions"][name]["text_correct"] for r in group)
                         for name in MODELS}}
    result["by_catalog"]=per_catalog
    result["by_codec"]=per_codec
    return result


def main(args):
    verify_commit(args.source,VIM_REV)
    verify_commit(args.training,TRAIN_REV)
    if chardet.__version__!="7.7.0" or charset_normalizer.__version__!="3.5.2":
        raise RuntimeError("pin both comparator versions before scoring")
    origin=get_frozen(args.source,args.manifest)
    training=[(raw,label,group) for raw,label,group,k,h in corpus(pathlib.Path(args.training))]
    # We must not leak exact decoded content even across encoding variants.
    training_text_ids={sha(strict_text(raw,label).encode("utf-8","surrogatepass"))
                       for raw,label,group in training}
    seen=set()
    kept=[];drops=collections.Counter()
    for row in origin:
        ident=sha(row["text"].encode("utf-8","surrogatepass"))
        if ident in training_text_ids:
            drops["training_content_duplicate"]+=1
            continue
        if row["sha"] in seen:
            drops["duplicate_raw_window"]+=1
            continue
        seen.add(row["sha"])
        kept.append(row)
    if len(kept)<70:
        raise RuntimeError(f"insufficient independent real-source byte windows: {len(kept)}")
    residual,_=residual_subset(training)
    generated,_=contrastive_augmentation(residual,per_direction=24,exclude_ebcdic=True)
    baseline=TrainableDetector.fit(residual,feature_profile="stable23")
    specialist=TrainableDetector.fit(residual+generated,feature_profile="stable23")
    model_set={"stable23":baseline,
               "latin_specialist":specialist,
               "selective_router":SelectiveLatinDetector(baseline,specialist)}
    results=[]
    for item in kept:
        raw=item["raw"];correct=item["text"];codec=item["codec"]
        preds={}
        for name,model in model_set.items():
            answer=decode(model,raw,bom_policy="preserve")
            try: label_hit=bool(answer.encoding and codecs.lookup(answer.encoding).name==codec)
            except LookupError: label_hit=False
            preds[name]={"label":answer.encoding,"text_correct":answer.text==correct,
                         "label_correct":label_hit,"abstained":answer.text is None,
                         "ambiguous":answer.ambiguous}
        for name in ("chardet7","charset_normalizer"):
            label,text=third_party(name,raw)
            try:label_hit=bool(label and codecs.lookup(label).name==codec)
            except LookupError:label_hit=False
            preds[name]={"label":label,"text_correct":text==correct,
                         "label_correct":label_hit,"abstained":text is None}
        results.append({"file":item["file"],"window":item["window"],"codec":codec,
                        "sha256":item["sha"],"generated_by_build":item["generated"],
                        "predictions":preds})
    s=stats(results)
    report={
        "schema":"codeccat-vim-real-byte-holdout-v1",
        "status":"ONE_SHOT_CONSUMED; NO_AUTOMATIC_RELEASE",
        "source":"vim/vim source-controlled PO files; translation-only original bytes",
        "frozen_source_run":"https://github.com/Johnny-Kao/CodecCat/actions/runs/41238184283",
        "frozen_manifest_fingerprint":FINGERPRINT,
        "source_revision":VIM_REV,
        "n_frozen":len(origin),"n_evaluated":len(kept),
        "excluded":dict(drops),
        "metrics":"exact strict Unicode text from declared PO charset; codec label separately",
        "by_source_windows_correlated":True,
        "generated_catalog_windows":sum(x["generated_by_build"] for x in results),
        "models":{"train_revision":TRAIN_REV,"n_base":len(residual),
                  "n_specialist":len(residual)+len(generated),
                  "added_specialist_training":len(generated)},
        "outcomes":s,
        "warnings":[
            "Multiple windows from one catalog are correlated and are not i.i.d. test documents",
            "Some source-controlled PO catalogs are build-generated variants of other catalogs",
            "Extracted msgstr bytes retain literal gettext escapes but contain no charset header",
            "Not evaluated for population-prevalence weighting or production reliability",
            "No native test content committed to CodecCat and no test bytes used in fit",
        ],
    }
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    (out/"paired.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True) for x in results)+"\n")
    print("VIM_REAL_BYTE_RESULT_BEGIN")
    print(json.dumps({k:report[k] for k in ("n_frozen","n_evaluated","excluded","generated_catalog_windows","outcomes")},
                     sort_keys=True,indent=2))
    print("VIM_REAL_BYTE_RESULT_END")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--training",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",default="research-results/vim-po-one-shot")
    main(p.parse_args())
