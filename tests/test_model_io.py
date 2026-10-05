from __future__ import annotations

import numpy as np

from codeccat import LinearModel, RuntimeBundle, load_bundle, save_bundle


def _model(classes=("a", "b"), width=3):
    return LinearModel(
        classes=classes,
        weight=np.arange(max(1, len(classes) - 1) * width, dtype=np.float64).reshape(max(1, len(classes) - 1), width),
        bias=np.zeros(max(1, len(classes) - 1), dtype=np.float64),
    )


def test_bundle_roundtrip(tmp_path):
    route = _model(width=3)
    bundle = RuntimeBundle(
        route_models={"U": route, "N": route, "RL": route, "RH": route},
        calibrator=None,
        triad=None,
        utf8_sig=None,
        utf8_gb=None,
        classes=("a", "b"),
        families=("other",),
    )
    path = tmp_path / "model.npz"
    save_bundle(path, bundle)
    loaded = load_bundle(path)
    assert loaded.classes == bundle.classes
    assert set(loaded.route_models) == {"U", "N", "RL", "RH"}
    np.testing.assert_array_equal(loaded.route_models["U"].weight, route.weight)
