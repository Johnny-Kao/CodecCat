"""Opt-in legacy API smoke test (requires codeccat[legacy])."""
import numpy as np

import codeccat
from codeccat import (
    DetectionResult, Detector, LinearModel, RuntimeBundle, RuntimeStack,
    load_bundle, save_bundle,
)
from codeccat.api import DetectionResult as PriorDetectionResult
from codeccat.model import RuntimeBundle as PriorRuntimeBundle

assert DetectionResult is PriorDetectionResult
assert RuntimeBundle is PriorRuntimeBundle
assert DetectionResult("utf-8", ("utf-8",)).encoding == "utf-8"
m = LinearModel(("utf-8", "cp1252"), np.zeros((2, 518)), np.zeros((2,)))
stack = RuntimeStack(
    {name: m for name in ("U", "N", "RL", "RH")},
    None, None, None, None,
)
bundle = RuntimeBundle(
    baseline=stack,
    full_b=None,
    rl_specialist=None,
    gate_thresholds={},
    classes=("utf-8", "cp1252"),
    families=("utf", "singlebyte-win"),
)
assert bundle.route_models is stack.route_models
assert Detector(bundle).__class__ is Detector
assert callable(save_bundle) and callable(load_bundle)
assert codeccat.detect(b"hello").text == "hello"
print("RELEASE_LEGACY_API_PASS", {"symbols": 7, "numpy_optional": True})
