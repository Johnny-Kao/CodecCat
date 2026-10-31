"""Reproducible yearly candidate training. Does not promote or publish models.

The held-out grouped fold experiment must run independently before any release
decision; this step packages candidates plus precise lineage for review.
"""
import argparse,codecs,hashlib,json,platform,subprocess,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy,sklearn
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_l3_language import LanguageCoherence
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus

CONFIG_VERSION="annual-v1"
PROFILES=("stable23","experimental13")
def manifest_rows(rows):
    h=hashlib.sha256()
    for b,label,group,fold,sha in rows:
        h.update(json.dumps([sha,label,group,fold],separators=(",",":")).encode()+b"\n")
    return h.hexdigest()
def main(a):
    if not (2026<=a.year<=2100):raise ValueError("year outside supported window")
    # The manifest must describe the bytes ACTUALLY checked out, not an
    # unverified caller-provided commit string.
    resolved=subprocess.run(
        ["git","-C",str(Path(a.training)),"rev-parse","HEAD"],
        capture_output=True,text=True,check=True,
    ).stdout.strip().lower()
    if resolved != a.training_commit.lower():
        raise ValueError("training source commit does not match checked-out HEAD")
    rows=corpus(Path(a.training))
    if len(rows)<100:raise ValueError("too few labeled samples to train")
    groups=sorted({r[2] for r in rows})
    folds=[{r[2] for r in rows if r[3]==k} for k in range(3)]
    if any(folds[i]&folds[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("source-group leaked across folds")
    if len(set(r[4] for r in rows)) != len(rows):
        raise ValueError("duplicate bytes in accepted training records")
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    items=[(b,label,group) for b,label,group,fold,sha in rows]
    report={"recipe":CONFIG_VERSION,"year":a.year,"created_utc":datetime.now(timezone.utc).isoformat(),
      "training_origin":"chardet/test-data","training_commit":a.training_commit,
      "dataset_fingerprint":manifest_rows(rows),"samples":len(rows),"source_groups":len(groups),
      "label_counts":dict(sorted(Counter(r[1] for r in rows).items())),
      "folds":[{"fold":k,"samples":sum(x[3]==k for x in rows),
                "source_groups":len(folds[k])} for k in range(3)],
      "dependencies":{"python":platform.python_version(),"numpy":numpy.__version__,"scikit_learn":sklearn.__version__},
      "profiles":{},"release_authorized":False,
      "requirements":"Grouped CV plus blind fresh-source evaluation, paired regression and resource checks required before promotion."}
    for profile in PROFILES:
        model=TrainableDetector.fit(items,feature_profile=profile)
        name=f"codeccat-{a.year}-{profile}.zip"
        model.save_portable(dest/name)
        again=TrainableDetector.load_portable(dest/name)
        probes=[rows[i][0] for i in sorted({0,len(rows)//4,len(rows)//2,3*len(rows)//4,len(rows)-1})]
        if any(model.rank(x).encoding!=again.rank(x).encoding for x in probes):
            raise RuntimeError(f"round-trip mismatch for {profile}")
        report["profiles"][profile]={"archive":name,"sha256":hashlib.sha256((dest/name).read_bytes()).hexdigest(),
            "fingerprint":model.fingerprint,"vocab":len(model.vectorizer.vocabulary_),
            "class_count":len(model.classifier.classes_),"bytes":(dest/name).stat().st_size,
            "ngram_range":list(model.vectorizer.ngram_range)}
        print("ANNUAL_PROFILE_READY",profile,flush=True)
    # A separately fitted Unicode language model is packaged as an OPTIONAL
    # candidate; it does not change the stable default or release authority.
    l3 = LanguageCoherence.fit(items)
    l3_name = f"codeccat-{a.year}-language-l3.json"
    l3.save(dest/l3_name)
    if LanguageCoherence.load(dest/l3_name) != l3:
        raise RuntimeError("L3 portable round-trip mismatch")
    report["language_l3"] = {
        "archive": l3_name,
        "sha256": hashlib.sha256((dest/l3_name).read_bytes()).hexdigest(),
        "fingerprint": l3.fingerprint,
        "feature_count": len(l3.grams),
        "bytes": (dest/l3_name).stat().st_size,
        "default_enabled": False,
        "status": "experimental, requires source-independent net benefit",
    }
    print("ANNUAL_LANGUAGE_L3_CANDIDATE_READY", flush=True)
    (dest/"manifest.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("ANNUAL_MANIFEST_BEGIN");print(json.dumps(report,indent=2));print("ANNUAL_MANIFEST_END")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--training",required=True)
    p.add_argument("--training-commit",required=True)
    p.add_argument("--output",default="research-results/vnext-annual-models")
    main(p.parse_args())
