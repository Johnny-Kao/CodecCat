from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import LinearModel, RuntimeBundle, RuntimeStack


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


def _put_stack(arrays: dict[str, np.ndarray], meta: dict, prefix: str, stack: RuntimeStack | None) -> None:
    if stack is None:
        meta[prefix] = None
        return
    meta[prefix] = {"routes": list(stack.route_models)}
    for route, model in stack.route_models.items():
        _put_model(arrays, meta, f"{prefix}.route.{route}", model)
    _put_model(arrays, meta, f"{prefix}.calibrator", stack.calibrator)
    _put_model(arrays, meta, f"{prefix}.triad", stack.triad)
    _put_model(arrays, meta, f"{prefix}.utf8_sig", stack.utf8_sig)
    _put_model(arrays, meta, f"{prefix}.utf8_gb", stack.utf8_gb)


def _get_stack(data, meta: dict, prefix: str) -> RuntimeStack | None:
    info = meta[prefix]
    if info is None:
        return None
    route_models = {
        route: _get_model(data, meta, f"{prefix}.route.{route}")
        for route in info["routes"]
    }
    if any(model is None for model in route_models.values()):
        raise ValueError(f"{prefix} route model cannot be null")
    return RuntimeStack(
        route_models=route_models,  # type: ignore[arg-type]
        calibrator=_get_model(data, meta, f"{prefix}.calibrator"),
        triad=_get_model(data, meta, f"{prefix}.triad"),
        utf8_sig=_get_model(data, meta, f"{prefix}.utf8_sig"),
        utf8_gb=_get_model(data, meta, f"{prefix}.utf8_gb"),
    )


def save_bundle(path: str | Path, bundle: RuntimeBundle) -> None:
    arrays: dict[str, np.ndarray] = {}
    meta = {
        "schema": 2,
        "classes": list(bundle.classes),
        "families": list(bundle.families),
        "gate_thresholds": {k: float(v) for k, v in bundle.gate_thresholds.items()},
        "r12_threshold": float(bundle.r12_threshold),
    }
    _put_stack(arrays, meta, "baseline", bundle.baseline)
    _put_stack(arrays, meta, "full_b", bundle.full_b)
    _put_model(arrays, meta, "rl_specialist", bundle.rl_specialist)
    arrays["__manifest__"] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez_compressed(Path(path), **arrays)


def _load_schema1(data, meta: dict) -> RuntimeBundle:
    route_models = {
        route: _get_model(data, meta, f"route.{route}")
        for route in meta["routes"]
    }
    if any(model is None for model in route_models.values()):
        raise ValueError("route model cannot be null")
    baseline = RuntimeStack(
        route_models=route_models,  # type: ignore[arg-type]
        calibrator=_get_model(data, meta, "calibrator"),
        triad=_get_model(data, meta, "triad"),
        utf8_sig=_get_model(data, meta, "utf8_sig"),
        utf8_gb=_get_model(data, meta, "utf8_gb"),
    )
    return RuntimeBundle(
        baseline=baseline,
        full_b=None,
        rl_specialist=None,
        gate_thresholds={},
        classes=tuple(meta["classes"]),
        families=tuple(meta["families"]),
    )


def load_bundle(path: str | Path) -> RuntimeBundle:
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["__manifest__"]))
        schema = meta.get("schema")
        if schema == 1:
            return _load_schema1(data, meta)
        if schema != 2:
            raise ValueError(f"unsupported CodecCat model schema: {schema}")

        baseline = _get_stack(data, meta, "baseline")
        if baseline is None:
            raise ValueError("baseline stack cannot be null")
        return RuntimeBundle(
            baseline=baseline,
            full_b=_get_stack(data, meta, "full_b"),
            rl_specialist=_get_model(data, meta, "rl_specialist"),
            gate_thresholds={k: float(v) for k, v in meta.get("gate_thresholds", {}).items()},
            classes=tuple(meta["classes"]),
            families=tuple(meta["families"]),
            r12_threshold=float(meta.get("r12_threshold", 0.65)),
        )
