from __future__ import annotations

import numpy as np

from codeccat import LinearModel, RuntimeBundle, RuntimeStack, load_bundle, save_bundle


def _model(classes=("a", "b"), width=3):
    return LinearModel(
        classes=classes,
        weight=np.arange(max(1, len(classes) - 1) * width, dtype=np.float64).reshape(max(1, len(classes) - 1), width),
        bias=np.zeros(max(1, len(classes) - 1), dtype=np.float64),
    )


def _stack(width=3):
    route = _model(width=width)
    return RuntimeStack(
        route_models={"U": route, "N": route, "RL": route, "RH": route},
        calibrator=None,
        triad=None,
        utf8_sig=None,
        utf8_gb=None,
    )


def test_bundle_roundtrip_schema2(tmp_path):
    baseline = _stack(width=3)
    full_b = _stack(width=4)
    specialist = _model(classes=("utf-8", "__other__"), width=19)
    bundle = RuntimeBundle(
        baseline=baseline,
        full_b=full_b,
        rl_specialist=specialist,
        gate_thresholds={"U": 1.25, "RH": 2.5},
        classes=("a", "b"),
        families=("other",),
        r12_threshold=0.65,
    )
    path = tmp_path / "model.npz"
    save_bundle(path, bundle)
    loaded = load_bundle(path)
    assert loaded.classes == bundle.classes
    assert set(loaded.route_models) == {"U", "N", "RL", "RH"}
    assert loaded.gate_thresholds == {"U": 1.25, "RH": 2.5}
    assert loaded.r12_threshold == 0.65
    assert loaded.full_b is not None
    assert loaded.rl_specialist is not None
    np.testing.assert_array_equal(loaded.route_models["U"].weight, baseline.route_models["U"].weight)
    np.testing.assert_array_equal(loaded.full_b.route_models["RH"].weight, full_b.route_models["RH"].weight)
