"""Installable, model-backed public vNext API.

The packaged model is a reproducible numeric-only snapshot, not executable
pickle data. L3 remains explicitly opt-in because cross-source improvements
are not established and activation has measurable cost.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vnext_strong_l2 import TrainableDetector
    from .vnext_l3_language import LanguageCoherence
    from .vnext_decoding import Detection, BomPolicy


@lru_cache(maxsize=1)
def _packaged_model():
    from .vnext_strong_l2 import TrainableDetector
    resource = files("codeccat").joinpath("models/stable23.zip")
    if not resource.is_file():
        raise RuntimeError(
            "CodecCat's stable23 model was not included in this installation. "
            "Install an official model-bearing wheel or use CodecCat.from_files(...)."
        )
    with as_file(resource) as file_path:
        return TrainableDetector.load_portable(file_path)


@lru_cache(maxsize=1)
def _packaged_language_model():
    from .vnext_l3_language import LanguageCoherence
    resource = files("codeccat").joinpath("models/language-l3.json")
    if not resource.is_file():
        raise RuntimeError("The optional L3 language model is missing from this installation.")
    with as_file(resource) as file_path:
        return LanguageCoherence.load(file_path)


def _bytes(data: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("CodecCat expects a bytes-like input, not text")
    return bytes(data)


class CodecCat:
    """Detect and strictly decode one complete bytes object.

    Example:
        >>> result = CodecCat().detect(b"hello")
        >>> result.text
        'hello'

    This is not a streaming detector. Encoding labels can be ambiguous even
    when decoded text is unambiguous. Inspect result.ambiguous.
    """

    def __init__(
        self,
        model: TrainableDetector | None = None,
        *,
        language_model: LanguageCoherence | None = None,
        use_l3: bool = False,
    ):
        if language_model is not None and not use_l3:
            raise ValueError("pass use_l3=True to enable an explicit language_model")
        self._model = model if model is not None else _packaged_model()
        self._language_model = (
            language_model if language_model is not None else _packaged_language_model()
        ) if use_l3 else None

    @classmethod
    def from_files(
        cls,
        model_path: str | Path,
        *,
        language_path: str | Path | None = None,
    ) -> "CodecCat":
        """Use a local, numeric-only L2 model; L3 remains opt-in.

        Do not load arbitrary third-party models as trusted authoritative data.
        """
        from .vnext_strong_l2 import TrainableDetector
        from .vnext_l3_language import LanguageCoherence
        model = TrainableDetector.load_portable(model_path)
        language = LanguageCoherence.load(language_path) if language_path is not None else None
        return cls(model, language_model=language, use_l3=language is not None)

    @property
    def model_fingerprint(self) -> str:
        return self._model.fingerprint

    def detect(
        self,
        data: bytes | bytearray | memoryview,
        *,
        bom_policy: BomPolicy = "strip",
    ) -> Detection:
        from .vnext_decoding import decode as _decode
        return _decode(
            self._model, _bytes(data),
            bom_policy=bom_policy,
            language_model=self._language_model,
        )

    def decode(
        self,
        data: bytes | bytearray | memoryview,
        *,
        bom_policy: BomPolicy = "strip",
    ) -> str | None:
        """Return strict decoded text or None if no valid selection was found."""
        return self.detect(data, bom_policy=bom_policy).text


@lru_cache(maxsize=1)
def _default_detector() -> CodecCat:
    return CodecCat()


def detect(
    data: bytes | bytearray | memoryview, *, bom_policy: BomPolicy = "strip",
) -> Detection:
    """One-call detection using the bundled stable23 model (L3 disabled)."""
    return _default_detector().detect(data, bom_policy=bom_policy)


def decode(
    data: bytes | bytearray | memoryview, *, bom_policy: BomPolicy = "strip",
) -> str | None:
    """One-call strict decoding; returns None instead of lossy replacement."""
    return detect(data, bom_policy=bom_policy).text
