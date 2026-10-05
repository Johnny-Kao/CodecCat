from __future__ import annotations

import numpy as np

from codeccat.features import feature_vector, hmt768, route


def test_hmt768_is_bounded():
    data = bytes(range(256)) * 8
    sampled = hmt768(data)
    assert len(sampled) == 768
    assert sampled[:256] == data[:256]
    assert sampled[-256:] == data[-256:]


def test_feature_shape_and_dtype():
    vector = feature_vector(b"hello world")
    assert vector.shape == (518,)
    assert vector.dtype == np.float32


def test_route_utf8_and_nul():
    assert route("hello".encode()) == "U"
    assert route(b"\xff\x00\xfe") == "N"


def test_route_residual_split():
    assert route(bytes([0x80]) + b"a" * 100) == "RL"
    assert route(bytes([0x80]) * 20 + b"a" * 20) == "RH"
