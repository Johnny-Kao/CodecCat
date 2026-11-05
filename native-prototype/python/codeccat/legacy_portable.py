"""Optional legacy stable23/ordered13 numeric ZIP model adapter.

The original format stores JSON metadata plus compressed NumPy arrays, never
pickle. Loading it requires only the opt-in NumPy dependency. Inference still
uses the Rust-native L2 scorer; sklearn is not imported.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

SCHEMA_23 = "vnext-l2-tfidf-ordered23-sgd-huber-v1"
SCHEMA_13 = "vnext-l2-tfidf-ordered13-sgd-huber-v2"


def load_legacy_zip(path: str | Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Legacy CodecCat .zip models require optional NumPy support; "
            "install 'codeccat[legacy]' or export your model to .ccm."
        ) from exc
    from ._codeccat_native import NativeL2
    from .vnext_strong_l2 import TrainableDetector

    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if sorted(names) != ["manifest.json", "weights.npz"]:
            raise ValueError("unexpected or duplicated model archive entries")
        manifest_info = archive.getinfo("manifest.json")
        tensor_info = archive.getinfo("weights.npz")
        if (manifest_info.file_size > 8_000_000 or
                tensor_info.file_size > 200_000_000):
            raise ValueError("legacy archive sections exceed maximum size")
        manifest = json.loads(archive.read("manifest.json"))
        schema = manifest.get("schema")
        expected_range = {
            SCHEMA_23: [2, 3],
            SCHEMA_13: [1, 3],
        }.get(schema)
        if expected_range is None:
            raise ValueError("unsupported legacy model schema")
        if manifest.get("params") != {
            "ngram_range": expected_range,
            "sublinear_tf": True,
            "lowercase": True,
            "norm": "l2",
        }:
            raise ValueError("unsupported legacy feature transform")

        with np.load(io.BytesIO(archive.read("weights.npz")),
                     allow_pickle=False) as tensors:
            if set(tensors.files) != {"idf", "coef", "intercept"}:
                raise ValueError("unexpected numeric tensor keys")
            idf = tensors["idf"].copy()
            coef = tensors["coef"].copy()
            intercept = tensors["intercept"].copy()

    vocab = manifest.get("vocabulary")
    classes = manifest.get("classes")
    fingerprint = manifest.get("fingerprint")
    if (not isinstance(vocab, dict) or not vocab or
            not isinstance(classes, list) or len(classes) < 2 or
            not all(isinstance(c, str) for c in classes) or
            not isinstance(fingerprint, str) or len(fingerprint) != 64 or
            sorted(vocab.values()) != list(range(len(vocab))) or
            idf.shape != (len(vocab),) or
            coef.shape != (1 if len(classes) == 2 else len(classes), len(vocab)) or
            intercept.shape != (coef.shape[0],) or
            not np.isfinite(idf).all() or not np.isfinite(coef).all() or
            not np.isfinite(intercept).all()):
        raise ValueError("invalid legacy model metadata or weights")
    native = NativeL2(vocab, idf.tolist(), coef.tolist(), intercept.tolist(),
                      expected_range[0], expected_range[1], False, True)
    return TrainableDetector(native, fingerprint, classes=classes)
