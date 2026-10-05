from __future__ import annotations

import math

import numpy as np

S3_THRESHOLD = 0.02


def hmt768(data: bytes) -> bytes:
    if len(data) <= 768:
        return data
    middle = len(data) // 2
    return data[:256] + data[middle - 128 : middle + 128] + data[-256:]


def strict_utf8(data: bytes) -> bool:
    try:
        data[:4096].decode("utf-8", "strict")
        return True
    except UnicodeDecodeError:
        return False


def has_utf8_bom(data: bytes) -> bool:
    return data.startswith(b"\xef\xbb\xbf")


def route(data: bytes) -> str:
    sample = data[:4096]
    if strict_utf8(sample):
        return "U"
    if b"\x00" in sample:
        return "N"
    array = np.frombuffer(sample, dtype=np.uint8)
    high_ratio = float(np.mean(array >= 128)) if len(array) else 0.0
    return "RL" if high_ratio <= S3_THRESHOLD else "RH"


def replacement_rate(data: bytes) -> float:
    if not data:
        return 0.0
    text = data.decode("utf-8", errors="replace")
    return text.count("\ufffd") / max(1, len(text))


def byte_signals(data: bytes) -> tuple[bool, bool, bool, float, float, int]:
    sample = data[:4096]
    array = np.frombuffer(sample, dtype=np.uint8)
    strict = strict_utf8(data)
    bom = has_utf8_bom(data)
    if len(array):
        high_count = int(np.count_nonzero(array >= 128))
        nul_count = int(np.count_nonzero(array == 0))
        high = high_count / len(array)
        nul = nul_count / len(array)
        ascii_only = high_count == 0
    else:
        high = 0.0
        nul = 0.0
        ascii_only = True
    return bom, strict, ascii_only, high, nul, len(array)


_REDUCE_IDX = np.asarray([32, 127, 128], dtype=np.intp)


def feature_vector(data: bytes) -> np.ndarray:
    sampled = hmt768(data)
    array = np.frombuffer(sampled, dtype=np.uint8)
    n = max(1, len(array))

    counts = np.bincount(array, minlength=256)
    out = np.empty(518, dtype=np.float32)
    out[:256] = counts
    out[:256] *= 1.0 / n

    if len(array) >= 2:
        # uint8 addition intentionally wraps modulo 256.
        bins = array[:-1] + array[1:]
        bigram_counts = np.bincount(bins, minlength=256)
        out[256:512] = bigram_counts
        out[256:512] *= 1.0 / (len(array) - 1)
    else:
        out[256:512] = 0.0

    if len(array):
        inv = 1.0 / len(array)
        aggregate = np.add.reduceat(counts, _REDUCE_IDX)
        printable = float(aggregate[0] * inv)
        high = float(aggregate[2] * inv)
        nul = float(counts[0] * inv)
        lf = float(counts[10] * inv)
        cr = float(counts[13] * inv)
    else:
        high = nul = printable = lf = cr = 0.0

    out[512:] = (
        math.log2(n + 1) / 16.0,
        high,
        nul,
        printable,
        lf,
        cr,
    )
    return out
