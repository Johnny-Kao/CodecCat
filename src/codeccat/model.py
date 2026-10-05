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
class RuntimeBundle:
    route_models: Mapping[str, LinearModel]
    calibrator: LinearModel | None
    triad: LinearModel | None
    utf8_sig: LinearModel | None
    utf8_gb: LinearModel | None
    classes: tuple[str, ...]
    families: tuple[str, ...]

    def __post_init__(self) -> None:
        missing = {"U", "N", "RL", "RH"} - set(self.route_models)
        if missing:
            raise ValueError(f"missing route models: {sorted(missing)}")
