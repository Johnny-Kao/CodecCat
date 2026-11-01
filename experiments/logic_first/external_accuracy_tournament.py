"""One-shot, source-separated external accuracy tournament (development only).

Use original fixture bytes; labels from filenames are not provenance attestation.
No production weights, feature extraction, or model-promotion changes.
"""
from __future__ import annotations
import argparse, codecs, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector

def decode_prefix(data, encoding):
    if not encoding:return None
    try:
        d=codecs.getincrementaldecoder(encoding)("strict")
        return d.decode(data[:4096], final=len(data)<4096)
    except (UnicodeError, LookupError, ValueError, TypeError):
        return None

def canonical(name):
    try:return codecs.lookup(name).name
    except (LookupError, ValueError):return None

def truth_label(path):
    # uchardet fixture convention is encoding name in file basename.
    if path.name == "ascii.txt":return "ascii"
    return canonical(path.stem.replace("_","-"))

def measure(truth, raw, prediction):
    actual=decode_prefix(raw,prediction)
    expected=decode_prefix(raw,truth)
    if expected is None:return None
    if actual is not None:
        # Uniform treatment of leading Unicode BOM, independent of codec alias.
        actual=actual.lstrip("\ufeff")
    expected=expected.lstrip("\ufeff")
    return actual==expected

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--training",required=True)
    p.add_argument("--external",required=True)
    p.add_argument("--out",default="research-results/external-batch-tournament")
    args=p.parse_args()
    rows=corpus(Path(args.training))
    training=[(b,y,g) for b,y,g,_,_ in rows]
    residual,_=residual_subset(training)
    print("FIT_STABLE23_RESIDUAL",len(residual),flush=True)
    model=TrainableDetector.fit(residual,feature_profile="stable23")
    import chardet
    from charset_normalizer import from_bytes
    known={hashlib.sha256(b).hexdigest() for b,_,_,_,_ in rows}
    cases=[]; skipped=[]; dup=defaultdict(list)
    for path in sorted((Path(args.external)/"test").rglob("*")):
        if not path.is_file() or path.name=="CMakeLists.txt":continue
        label=truth_label(path)
        if not label:
            skipped.append({"path":path.relative_to(args.external).as_posix(),"reason":"unknown Python codec"})
            continue
        raw=path.read_bytes()
        if not raw:continue
        digest=hashlib.sha256(raw).hexdigest()
        dup[digest].append(path.relative_to(args.external).as_posix())
        try:
            truth=decode_prefix(raw,label)
            if truth is None:raise UnicodeError("strict decoding fails")
        except (UnicodeError,LookupError):
            skipped.append({"path":path.relative_to(args.external).as_posix(),"reason":"invalid strict truth"})
            continue
        rank=model.rank(raw,final=len(raw)<=4096)
        pred=canonical(str(rank.encoding)) if rank.encoding else None
        try:competitor=canonical(chardet.detect(raw)["encoding"]) if chardet.detect(raw)["encoding"] else None
        except Exception:competitor=None
        try:
            best=from_bytes(raw).best()
            normalizer=canonical(best.encoding) if best else None
        except Exception:normalizer=None
        entry={"path":path.relative_to(args.external).as_posix(),
               "sha256":digest,"bytes":len(raw),"label":label,
               "exact_training_byte_overlap":digest in known,
               "same_text":{}, "label_match":{}, "predicted":{},
               "candidate_top3":list(rank.candidates)}
        for key,guess in [("codeccat",pred),("chardet",competitor),("charset_normalizer",normalizer)]:
            entry["predicted"][key]=guess
            entry["label_match"][key]=(guess==label)
            entry["same_text"][key]=measure(label,raw,guess)
        cases.append(entry)
    clean=[x for x in cases if not x["exact_training_byte_overlap"]]
    def summarize(records):
        return {name:{"n":len(records),
          "same_text":sum(c["same_text"][name] is True for c in records),
          "label_match":sum(c["label_match"][name] for c in records),
          "undecodable":sum(c["same_text"][name] is None for c in records)}
          for name in ("codeccat","chardet","charset_normalizer")}
    families=defaultdict(list)
    for c in clean:families[c["path"].split("/")[1]].append(c)
    report={"schema":"codeccat-external-batch-v1",
        "status":"third-party-fixture-development-diagnostic-NOT-blind-acceptance",
        "training_rows":len(residual),"all_valid_fixture_rows":len(cases),
        "nonexact-overlap_rows":len(clean),
        "same_text_metrics":summarize(clean),
        "by_language":{k:summarize(v) for k,v in sorted(families.items())},
        "exact_training_overlap":[x["path"] for x in cases if x["exact_training_byte_overlap"]],
        "within_external_duplicate_groups":[x for x in dup.values() if len(x)>1],
        "skipped":skipped,
        "scope":["Fixture filename encodings not independently authenticated.",
                 "All sources from one fixture repository; language files may be re-encodings of same source.",
                 "Different byte encodings of same text and near duplicates across training sets not yet excluded.",
                 "Model fitted at runtime, not released serialized weights.",
                 "No performance or model promotion claim."]}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"cases.jsonl").write_text("\n".join(json.dumps(c,sort_keys=True) for c in cases)+"\n")
    print("EXTERNAL_BATCH_JSON_START",flush=True)
    print(json.dumps(report,indent=2),flush=True)
    print("EXTERNAL_BATCH_JSON_END",flush=True)
if __name__=="__main__":main()
