"""Compare a user-trained numeric model with the published CodecCat default.

Explicitly evaluates bytes using the public Runtime, never raw scikit-learn
labels. No source bytes or decoded text are copied into the report.
"""
from __future__ import annotations
import codecs
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def evaluate(manifest: Path, candidate: Path, report: Path,
             split: str = "test", baseline: Path | None = None) -> dict:
    # Optional dependency only for evaluation; offline fitting remains independent.
    try:
        from codeccat import CodecCat
    except ImportError as ex:
        raise RuntimeError(
            "Evaluation needs the public Runtime: install 'codeccat' "
            "in the evaluation environment"
        ) from ex

    if split not in ("test", "validation"):
        raise ValueError("evaluation split must be 'test' or 'validation'")
    base = manifest.resolve().parent
    records=[]
    hashes=set()
    for index,line in enumerate(manifest.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        try:
            r=json.loads(line)
        except json.JSONDecodeError as ex:
            raise ValueError(f"manifest line {index}: invalid JSON") from ex
        if r.get("split") != split:
            continue
        for key in ("path","encoding","group"):
            if not isinstance(r.get(key),str) or not r[key]:
                raise ValueError(f"line {index}: missing {key}")
        rel=Path(r["path"])
        path=(base/rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(base):
            raise ValueError(f"line {index}: path escapes manifest directory")
        raw=path.read_bytes()
        if not raw:
            raise ValueError(f"line {index}: empty input")
        try:
            truth=raw.decode(r["encoding"],"strict")
        except (UnicodeError,LookupError) as ex:
            raise ValueError(f"line {index}: invalid encoding or ground truth") from ex
        canonical=codecs.lookup(r["encoding"]).name
        digest=hashlib.sha256(raw).hexdigest()
        if digest in hashes:
            raise ValueError(f"line {index}: duplicated evaluation content")
        hashes.add(digest)
        records.append((raw,canonical,truth))
    if not records:
        raise ValueError(f"manifest has no '{split}' examples")

    new=CodecCat.from_files(candidate)
    old=CodecCat.from_files(baseline) if baseline else CodecCat()
    both={"candidate":{"n":0,"accepted":0,"correct_text":0,"correct_label":0},
          "baseline":{"n":0,"accepted":0,"correct_text":0,"correct_label":0}}
    by_codec=defaultdict(lambda:{
        "n":0,"candidate_correct_text":0,"baseline_correct_text":0,
        "helps":0,"harms":0})
    helps=harms=ties_correct=ties_wrong=0
    for raw,canonical,truth in records:
        a=new.detect(raw);b=old.detect(raw)
        a_ok=a.text==truth;b_ok=b.text==truth
        group=by_codec[canonical]
        group["n"]+=1
        group["candidate_correct_text"]+=int(a_ok)
        group["baseline_correct_text"]+=int(b_ok)
        group["helps"]+=int(a_ok and not b_ok)
        group["harms"]+=int(b_ok and not a_ok)
        for name,det,ok in (("candidate",a,a_ok),("baseline",b,b_ok)):
            val=both[name]
            val["n"]+=1
            val["accepted"]+=int(det.accepted)
            val["correct_text"]+=int(ok)
            val["correct_label"]+=int(det.encoding is not None
                                      and codecs.lookup(det.encoding).name==canonical)
        helps+=int(a_ok and not b_ok)
        harms+=int(b_ok and not a_ok)
        ties_correct+=int(a_ok and b_ok)
        ties_wrong+=int(not a_ok and not b_ok)

    output={
        "schema":"codeccat-model-comparison-v1",
        "evaluation_split":split,
        "warning":"Only a held-out source-disjoint test manifest is independent. This tool cannot prove absence of near-duplicate or training-source overlap.",
        "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "candidate_model_sha256":hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "candidate_fingerprint":new.model_fingerprint,
        "baseline_model_sha256":hashlib.sha256(baseline.read_bytes()).hexdigest() if baseline else None,
        "baseline_fingerprint":old.model_fingerprint,
        "results":both,
        "by_codec":dict(sorted(by_codec.items())),
        "paired":{"helps":helps,"harms":harms,"both_correct":ties_correct,
                  "both_wrong":ties_wrong},
    }
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return output
