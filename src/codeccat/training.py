"""Reproducible user-data trainer for the existing R21-compatible baseline schema.

This vNext MVP trains all four routes on user bytes, uses grouped explicit
train/validation splits, and exports a Runtime-loadable bundle. It does not
claim to reproduce the frozen R21 architecture or its accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .features import baseline_from_array, hmt768_array
from .model import LinearModel, RuntimeBundle, RuntimeStack
from .model_io import save_bundle
from .runtime import Runtime, _signals, family

ROUTES = ("U", "N", "RL", "RH")


def read_manifest(path: Path):
    root = path.parent.resolve()
    entries = []
    digests = {}
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if not all(k in r for k in ("path", "encoding", "split", "group")):
            raise ValueError(f"Line {line_no}: need path, encoding, split, group")
        if r["split"] not in ("train", "validation"):
            raise ValueError(f"Line {line_no}: split must be train or validation")
        rel = Path(r["path"])
        if rel.is_absolute() or not (root / rel).resolve().is_relative_to(root):
            raise ValueError(f"Line {line_no}: unsafe relative file path")
        data = (root / rel).read_bytes()
        if not data:
            raise ValueError(f"Line {line_no}: empty input")
        digest = hashlib.sha256(data).hexdigest()
        previous = digests.get(digest)
        if previous and previous != (r["encoding"],r["split"]):
            raise ValueError(f"Line {line_no}: conflicting label or cross-split duplicate")
        digests[digest] = (r["encoding"],r["split"])
        entries.append((data, r["encoding"], r["split"], str(r["group"]),digest))
    train = [r for r in entries if r[2] == "train"]
    val = [r for r in entries if r[2] == "validation"]
    if not train or not val:
        raise ValueError("Both train and validation are required")
    if set(r[3] for r in train) & set(r[3] for r in val):
        raise ValueError("Group leakage between train and validation")
    return entries


def fit(path: Path, out: Path, report_path: Path, seed: int = 2026):
    from sklearn.linear_model import LogisticRegression
    records = read_manifest(path)
    classes = tuple(sorted({r[1] for r in records if r[2] == "train"}))
    if len(classes) < 2:
        raise ValueError("At least two training encodings required")
    unknown = sorted({r[1] for r in records if r[2] == "validation"} - set(classes))
    if unknown:
        raise ValueError(f"Validation labels missing from training: {unknown}")
    # A single global fit supplies a fallback when a route has one or zero classes.
    train_records = [r for r in records if r[2] == "train"]
    x = np.stack([baseline_from_array(hmt768_array(r[0])) for r in train_records]).astype(np.float64)
    labels = np.array([r[1] for r in train_records])
    route_names = np.array([_signals(r[0]).route for r in train_records])
    def train_linear(mask):
        y = labels[mask]
        observed = tuple(sorted(set(y)))
        if len(observed) < 2:
            if not observed:
                return None
            # Constant vote: zero coefficients, high bias for represented class.
            return LinearModel(observed, np.zeros((1,x.shape[1]),dtype=np.float64),np.array([1.0]))
        model = LogisticRegression(C=1.0,max_iter=500,random_state=seed,solver="lbfgs")
        model.fit(x[mask], y)
        if len(model.classes_)==2:
            # _raw() converts one logistic logit to [-z,z], matching sklearn binary ranking.
            weight = np.asarray(model.coef_,dtype=np.float64)
            bias = np.asarray(model.intercept_,dtype=np.float64)
        else:
            weight = np.asarray(model.coef_,dtype=np.float64)
            bias = np.asarray(model.intercept_,dtype=np.float64)
        return LinearModel(tuple(str(z) for z in model.classes_),weight,bias)
    global_model=train_linear(np.ones(len(labels),dtype=bool))
    routes={}
    for route in ROUTES:
        model=train_linear(route_names==route)
        routes[route]=model if model is not None else global_model
    bundle=RuntimeBundle(baseline=RuntimeStack(routes,None,None,None,None),
          full_b=None,rl_specialist=None,gate_thresholds={},
          classes=classes,families=tuple(sorted({family(z) for z in classes})))
    out.parent.mkdir(parents=True,exist_ok=True)
    save_bundle(out,bundle)
    from .model_io import load_bundle
    rt=Runtime(load_bundle(out))
    per_route=defaultdict(lambda:Counter())
    per_label=defaultdict(lambda:Counter())
    validation=[r for r in records if r[2]=="validation"]
    for data, label, split, group, digest in validation:
        pred=rt.rank(data)[0]
        rr=per_route[_signals(data).route];rr["n"]+=1;rr["hits"]+=int(pred==label)
        ll=per_label[label];ll["n"]+=1;ll["hits"]+=int(pred==label)
    report={"schema":"vnext-mvp-v1","feature_schema":"hmt768-baseline-518",
        "dataset_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed":seed,"train_n":len(train_records),"validation_n":len(validation),
        "accuracy":sum(c["hits"] for c in per_route.values())/len(validation),
        "routes":{k:dict(v) for k,v in sorted(per_route.items())},
        "labels":{k:dict(v) for k,v in sorted(per_label.items())},
        "warning":"Baseline-only learner; no calibrator, triad, Full-B or specialists; not performance-equivalent to R21"}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path,required=True,help="JSONL manifest")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=2026)
    a=ap.parse_args()
    print(json.dumps(fit(a.dataset,a.output,a.report,a.seed),indent=2))


if __name__=="__main__":
    main()
