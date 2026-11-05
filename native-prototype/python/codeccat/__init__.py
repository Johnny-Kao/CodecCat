"""CodecCat native staging package (no NumPy or scikit-learn at runtime).

Python API remains unchanged from the public Alpha. Legacy R21 classes are
deliberately excluded from this experimental wheel.
"""
from __future__ import annotations
from .usable import CodecCat, detect, decode
from .vnext_decoding import Detection
__all__ = ["CodecCat", "detect", "decode", "Detection"]
