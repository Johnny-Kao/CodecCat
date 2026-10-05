from __future__ import annotations

from dataclasses import dataclass

from .model import RuntimeBundle
from .runtime import Runtime


@dataclass(frozen=True)
class DetectionResult:
    encoding: str | None
    alternatives: tuple[str, ...]


class Detector:
    """CodecCat runtime bound to one immutable fitted model bundle."""

    def __init__(self, bundle: RuntimeBundle):
        self._runtime = Runtime(bundle)

    def rank(self, data: bytes) -> tuple[str, ...]:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("CodecCat expects a bytes-like input")
        return self._runtime.rank(bytes(data))

    def detect(self, data: bytes) -> DetectionResult:
        ranked = self.rank(data)
        return DetectionResult(
            encoding=ranked[0] if ranked else None,
            alternatives=ranked,
        )
