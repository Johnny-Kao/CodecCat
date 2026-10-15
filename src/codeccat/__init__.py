"""CodecCat namespace with opt-in legacy API loading.

The vNext modules must not eagerly import the frozen legacy model.
Existing public symbols stay available by lazy attribute lookup.
"""
from __future__ import annotations

__all__ = [
    "DetectionResult", "Detector", "LinearModel", "RuntimeBundle",
    "RuntimeStack", "load_bundle", "save_bundle",
]

def __getattr__(name):
    if name in ("DetectionResult", "Detector"):
        from . import api
        return getattr(api, name)
    if name in ("LinearModel", "RuntimeBundle", "RuntimeStack"):
        from . import model
        return getattr(model, name)
    if name in ("load_bundle", "save_bundle"):
        from . import model_io
        return getattr(model_io, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
