"""Rust L2 numeric parity through separate train-side and runtime interpreters.

Generate with the legacy Alpha + sklearn training environment.
Verify with the native public Wheel in a fresh environment *without* NumPy.
"""
from __future__ import annotations
import argparse,base64,json,math
from pathlib import Path

GOLDEN=Path("native-prototype/l2-numeric-goldens.jsonl")
MODEL=Path("native-prototype/model/stable23.ccm")

def inputs():
    from vnext_actual_adapter import corpus
    for item in corpus(Path("training"))[::12]:yield item[0]
    yield from (
        b"",b"hello",b"a"*4096,b"hello\n\nworld",
        b"\xff\x80\x00\x81",bytes(range(256)),
        "日本語".encode("utf-8"),"日本語".encode("utf-16"),
        "\ufeffcafé".encode("utf-8"),
    )

def generate():
    import numpy as np
    from codeccat.vnext_strong_l2 import TrainableDetector
    model=TrainableDetector.load_portable("src/codeccat/models/stable23.zip")
    GOLDEN.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with GOLDEN.open("w",encoding="utf-8") as out:
        for raw in inputs():
            v=model.vectorizer.transform([raw[:4096].decode("latin1")])
            score=np.asarray(model.classifier.decision_function(v)).reshape(-1)
            if len(model.classifier.classes_)==2:
                score=np.asarray([-score[0],score[0]])
            out.write(json.dumps({"raw":base64.b64encode(raw).decode(),
                "scores":score.tolist()},separators=(",",":"))+"\n")
            n+=1
    print("NATIVE_NUMERIC_GOLDENS_GENERATED",n,flush=True)

def verify():
    from codeccat._codeccat_native import NativeL2
    model=NativeL2.from_model_bytes(MODEL.read_bytes())
    n=0;max_error=0.0
    with GOLDEN.open(encoding="utf-8") as stream:
        for line in stream:
            item=json.loads(line)
            raw=base64.b64decode(item["raw"],validate=True)
            actual=model.scores(raw)
            old=item["scores"]
            assert len(actual)==len(old)
            delta=max((abs(a-b) for a,b in zip(actual,old)),default=0.0)
            max_error=max(max_error,delta)
            if delta>1e-9:
                raise AssertionError(f"L2 score parity mismatch case {n}: {delta}")
            order=lambda values:sorted(range(len(values)),key=lambda i:(values[i],i),reverse=True)
            if order(old)!=order(actual):
                raise AssertionError(f"L2 rank parity mismatch case {n}")
            n+=1
    assert n>=260,n
    corrupted=bytearray(MODEL.read_bytes())
    corrupted[len(corrupted)//2]^=1
    try:NativeL2.from_model_bytes(bytes(corrupted))
    except ValueError:pass
    else:raise AssertionError("invalid checksum allowed")
    print("NATIVE_NUMERIC_PARITY_PASS",
          json.dumps({"cases":n,"max_abs_error":max_error,"rank_mismatches":0}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("generate","verify"))
    args=ap.parse_args()
    if args.mode=="generate":generate()
    else:verify()
