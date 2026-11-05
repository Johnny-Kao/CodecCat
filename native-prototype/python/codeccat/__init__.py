"""CodecCat public API with lazily loaded opt-in legacy R21 compatibility.

Default import/detect/decode never imports NumPy. Legacy symbols preserve the
previous API when the user explicitly installs codeccat[legacy].
"""
from __future__ import annotations

from .usable import CodecCat, detect, decode
from .vnext_decoding import Detection

__all__ = [
    "CodecCat", "detect", "decode", "Detection",
    "DetectionResult", "Detector", "LinearModel", "RuntimeBundle",
    "RuntimeStack", "load_bundle", "save_bundle",
]

_LEGACY = {
    "DetectionResult": "api",
    "Detector": "api",
    "LinearModel": "model",
    "RuntimeBundle": "model",
    "RuntimeStack": "model",
    "load_bundle": "model_io",
    "save_bundle": "model_io",
}


def __getattr__(name: str):
    mod = _LEGACY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from importlib import import_module
        module = import_module(f".{mod}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise ImportError(
                f"{name} is a legacy CodecCat API; "
                "install optional compatibility support with "
                "'pip install codeccat[legacy]'"
            ) from exc
        raise
    result = getattr(module, name)
    globals()[name] = result
    return result
