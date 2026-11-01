"""Local CodecCat model selection.

The bundled model is used until the library owner explicitly selects a staged
candidate. There is no identity service, approval token or background update.
This registry is local state owned by the caller.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

from .model_lifecycle import SCHEMA, _atomic_json, _load_state, _verify_model


@contextmanager
def _exclusive(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".state.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def current_model(root: str | Path, bundled_model: str | Path) -> Path:
    """Return the default model until a user explicitly installs a candidate."""
    root = Path(root)
    state = _load_state(root)
    if state["active"] is None:
        default = Path(bundled_model)
        if not default.is_file():
            raise FileNotFoundError("bundled default model unavailable")
        return default
    return _verify_model(root, state["active"])


def select_candidate(root: str | Path, digest: str, *, expected_generation: int) -> dict:
    """Explicit local update. Caller controls when to select the candidate."""
    if re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        raise ValueError("invalid candidate digest")
    if type(expected_generation) is not int or expected_generation < 0:
        raise ValueError("invalid expected generation")
    root = Path(root)
    with _exclusive(root):
        state = _load_state(root)
        if state["generation"] != expected_generation:
            raise ValueError("stale model generation")
        if state["active"] == digest:
            raise ValueError("candidate is already active")
        record = json.loads((root / "candidates" / (digest + ".json")).read_text())
        if record.get("model_sha256") != digest or record.get("gate", {}).get("accepted") is not True:
            raise ValueError("candidate failed validation")
        model_path = _verify_model(root, digest)
        # Verify saved model has the claimed training fingerprint.
        from .vnext_strong_l2 import TrainableDetector
        loaded = TrainableDetector.load_portable(model_path)
        if loaded.fingerprint != record.get("training_fingerprint"):
            raise ValueError("candidate fingerprint mismatch")
        if state["active"] is not None:
            _verify_model(root, state["active"])
        updated = {
            "schema": SCHEMA,
            "active": digest,
            "previous": state["active"],
            "generation": state["generation"] + 1,
        }
        _atomic_json(root / "active.json", updated)
        return updated


def rollback_model(root: str | Path, *, expected_generation: int) -> dict:
    """Explicit local rollback to the previously selected candidate."""
    root = Path(root)
    with _exclusive(root):
        state = _load_state(root)
        if state["generation"] != expected_generation:
            raise ValueError("stale model generation")
        if state["previous"] is None:
            raise ValueError("no previous candidate; use bundled default manually")
        _verify_model(root, state["active"])
        _verify_model(root, state["previous"])
        updated = {
            "schema": SCHEMA,
            "active": state["previous"],
            "previous": state["active"],
            "generation": state["generation"] + 1,
        }
        _atomic_json(root / "active.json", updated)
        return updated
