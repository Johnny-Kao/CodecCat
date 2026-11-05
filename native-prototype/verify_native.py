"""Compare native Rust TF-IDF scores against the frozen shipped stable23 model.

This script runs in a TRAINING/VERIFICATION environment with sklearn/NumPy.
The installed Rust wheel itself has no runtime dependencies.
"""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np
from codeccat.vnext_strong_l2 import TrainableDetector
from vnext_actual_adapter import corpus
from _codeccat_native import NativeL2

def make_native(model):
    vec=model.vectorizer
    clf=model.classifier
    assert vec.ngram_range in ((1,2),(2,3),(1,3))
    return NativeL2(
        vec.vocabulary_,
        vec.idf_.tolist(),
        clf.coef_.tolist(),
        clf.intercept_.tolist(),
        *vec.ngram_range,
        bool(vec.binary),bool(vec.sublinear_tf),
    )

def reference_scores(model,b):
    v=model.vectorizer.transform([b[:4096].decode("latin1")])
    raw=np.asarray(model.classifier.decision_function(v)).reshape(-1)
    if len(model.classifier.classes_)==2:return np.array([-raw[0],raw[0]])
    return raw

def main():
    model=TrainableDetector.load_portable("src/codeccat/models/stable23.zip")
    native=make_native(model)
    dataset=corpus(Path("training"))
    # Source-independent deterministic stride, plus Unicode/BOM/binary corner cases.
    rows=[item[0] for item in dataset[::12]]
    rows.extend([
        b"",b"hello",b"a"*4096,b"hello\n\nworld",
        b"\xff\x80\x00\x81",bytes(range(256)),
        "\u00b5\u00d8\u00e9".encode("latin1"),
        "日本語".encode("utf-8"),
        "日本語".encode("utf-16"),
        "\ufeffcafé".encode("utf-8"),
    ])
    max_err=0.0;ranking_fail=0;max_relative=0.0
    begin=time.perf_counter()
    for n,b in enumerate(rows):
        expected=reference_scores(model,b)
        actual=np.asarray(native.scores(b),dtype=np.float64)
        if expected.shape!=actual.shape:raise AssertionError((n,expected.shape,actual.shape))
        delta=float(np.max(np.abs(expected-actual)))
        max_err=max(max_err,delta)
        max_relative=max(max_relative,float(np.max(np.abs(expected-actual)/(1+np.abs(expected)))))
        if not np.array_equal(np.argsort(expected)[::-1],np.argsort(actual)[::-1]):
            ranking_fail+=1
        if delta>1e-9:
            raise AssertionError(f"Native L2 diverged on case {n}, length={len(b)}, max error={delta}")
    report={"model":"alpha stable23","cases":len(rows),
      "score_max_abs_error":max_err,"score_max_relative_error":max_relative,
      "ranking_mismatch_count":ranking_fail,
      "duration_s":round(time.perf_counter()-begin,3),
      "native_module":"_codeccat_native",
      "note":"Gate 1 numerical parity only. No full L1/L3 or end-user wheel parity claimed."}
    print("NATIVE_L2_PARITY_RESULT",json.dumps(report),flush=True)
    if ranking_fail:raise AssertionError("numeric rank ordering diverged")

if __name__=="__main__":main()
