from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class LinearModel:
    classes: tuple[str, ...]
    weight: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class RuntimeStack:
    route_models: Mapping[str, LinearModel]
    calibrator: LinearModel | None
    triad: LinearModel | None
    utf8_sig: LinearModel | None
    utf8_gb: LinearModel | None

    def __post_init__(self) -> None:
        missing = {"U", "N", "RL", "RH"} - set(self.route_models)
        if missing:
            raise ValueError(f"missing route models: {sorted(missing)}")


@dataclass(frozen=True)
class RuntimeBundle:
    baseline: RuntimeStack
    full_b: RuntimeStack | None
    rl_specialist: LinearModel | None
    gate_thresholds: Mapping[str, float]
    classes: tuple[str, ...]
    families: tuple[str, ...]
    r12_threshold: float = 0.65

    def __post_init__(self) -> None:
        if self.full_b is not None:
            for route in ("U", "RH"):
                if route not in self.full_b.route_models:
                    raise ValueError(f"full-B stack missing route: {route}")
        for route in self.gate_thresholds:
            if route not in ("U", "RH"):
                raise ValueError(f"unsupported gate route: {route}")

    @property
    def route_models(self) -> Mapping[str, LinearModel]:
        """Compatibility alias for the baseline route models."""
        return self.baseline.route_models

    @property
    def calibrator(self) -> LinearModel | None:
        return self.baseline.calibrator

    @property
    def triad(self) -> LinearModel | None:
        return self.baseline.triad

    @property
    def utf8_sig(self) -> LinearModel | None:
        return self.baseline.utf8_sig

    @property
    def utf8_gb(self) -> LinearModel | None:
        return self.baseline.utf8_gb
