from __future__ import annotations

import numpy as np

from codeccat.features import (
    baseline_from_array,
    feature_vector,
    full_b_extra_from_array,
    full_b_vector,
    hmt768,
    hmt768_array,
    route,
)


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


def test_cached_baseline_matches_public_feature_vector():
    data = bytes(range(256)) * 5
    arr = hmt768_array(data)
    np.testing.assert_array_equal(baseline_from_array(arr), feature_vector(data))


def test_full_b_shape_and_conditional_zero():
    data = b"hello world"
    full = full_b_vector(data, "U")
    assert full.shape == (886,)
    assert full.dtype == np.float32
    extra = full_b_extra_from_array(hmt768_array(data), "RL")
    assert extra.shape == (368,)
    assert not np.any(extra)


def test_route_utf8_and_nul():
    assert route("hello".encode()) == "U"
    assert route(b"\xff\x00\xfe") == "N"


def test_route_residual_split():
    assert route(bytes([0x80]) + b"a" * 100) == "RL"
    assert route(bytes([0x80]) * 20 + b"a" * 20) == "RH"
