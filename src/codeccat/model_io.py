from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import LinearModel, RuntimeBundle


def _put_model(arrays: dict[str, np.ndarray], meta: dict, prefix: str, model: LinearModel | None) -> None:
    if model is None:
        meta[prefix] = None
        return
    meta[prefix] = {"classes": list(model.classes)}
    arrays[f"{prefix}.weight"] = np.asarray(model.weight, dtype=np.float64)
    arrays[f"{prefix}.bias"] = np.asarray(model.bias, dtype=np.float64)


def _get_model(data, meta: dict, prefix: str) -> LinearModel | None:
    info = meta[prefix]
    if info is None:
        return None
    return LinearModel(
        classes=tuple(info["classes"]),
        weight=np.asarray(data[f"{prefix}.weight"], dtype=np.float64),
        bias=np.asarray(data[f"{prefix}.bias"], dtype=np.float64),
    )


def save_bundle(path: str | Path, bundle: RuntimeBundle) -> None:
    arrays: dict[str, np.ndarray] = {}
    meta = {
        "schema": 1,
        "classes": list(bundle.classes),
        "families": list(bundle.families),
        "routes": list(bundle.route_models),
    }
    for route, model in bundle.route_models.items():
        _put_model(arrays, meta, f"route.{route}", model)
    _put_model(arrays, meta, "calibrator", bundle.calibrator)
    _put_model(arrays, meta, "triad", bundle.triad)
    _put_model(arrays, meta, "utf8_sig", bundle.utf8_sig)
    _put_model(arrays, meta, "utf8_gb", bundle.utf8_gb)
    arrays["__manifest__"] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez_compressed(Path(path), **arrays)


def load_bundle(path: str | Path) -> RuntimeBundle:
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["__manifest__"]))
        if meta.get("schema") != 1:
            raise ValueError(f"unsupported CodecCat model schema: {meta.get('schema')}")
        route_models = {
            route: _get_model(data, meta, f"route.{route}")
            for route in meta["routes"]
        }
        if any(model is None for model in route_models.values()):
            raise ValueError("route model cannot be null")
        return RuntimeBundle(
            route_models=route_models,  # type: ignore[arg-type]
            calibrator=_get_model(data, meta, "calibrator"),
            triad=_get_model(data, meta, "triad"),
            utf8_sig=_get_model(data, meta, "utf8_sig"),
            utf8_gb=_get_model(data, meta, "utf8_gb"),
            classes=tuple(meta["classes"]),
            families=tuple(meta["families"]),
        )
