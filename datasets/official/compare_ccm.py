"""Read-only CCM2 numeric difference audit (never prints model vocabulary).

Only aggregate statistics and class labels are emitted, not raw training text
or original model files. The parser enforces file and tensor size bounds.
"""
from __future__ import annotations

import argparse
from array import array
from hashlib import sha256
import json
from math import isfinite, sqrt
from pathlib import Path
import struct
import sys

MAX_BYTES = 256 * 1024 * 1024
MAX_FEATURES = 200_000
MAX_CLASSES = 256


class Cursor:
    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.pos = 0

    def take(self, size: int) -> bytes:
        if size < 0 or self.pos + size > len(self.data):
            raise ValueError("truncated CCM2")
        block = self.data[self.pos:self.pos+size]
        self.pos += size
        return bytes(block)

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def string(self) -> str:
        length, = self.unpack("<H")
        if not 0 < length <= 256:
            raise ValueError("invalid CCM2 model string")
        return self.take(length).decode("utf-8")


def floats(reader: Cursor, n: int, code: str) -> array:
    size = 8 if code == "d" else 4
    result = array(code)
    result.frombytes(reader.take(n * size))
    if sys.byteorder != "little":
        result.byteswap()
    if not all(isfinite(x) for x in result):
        raise ValueError("non-finite CCM2 tensor")
    return result


def load_ccm2(path: Path):
    raw = path.read_bytes()
    if not 32 + 16 < len(raw) <= MAX_BYTES:
        raise ValueError("invalid CCM2 file size")
    body, digest = raw[:-32], raw[-32:]
    if sha256(body).digest() != digest:
        raise ValueError("CCM2 digest mismatch")
    reader = Cursor(body)
    magic, width, nclasses, gram_min, gram_max, flags, reserved = reader.unpack(
        "<4sIIBBBB"
    )
    if magic != b"CCM2" or not 0 < width <= MAX_FEATURES or not 2 <= nclasses <= MAX_CLASSES:
        raise ValueError("invalid CCM2 dimensions or format")
    if not (0 < gram_min <= gram_max <= 3 and flags & ~3 == 0 and reserved == 0):
        raise ValueError("invalid CCM2 feature profile")
    classes = [reader.string() for _ in range(nclasses)]
    vocabulary_hash = sha256()
    for _ in range(width):
        index, = reader.unpack("<I")
        gram = reader.string()
        if not 0 <= index < width or not gram_min <= len(gram) <= gram_max:
            raise ValueError("invalid CCM2 vocabulary")
        vocabulary_hash.update(struct.pack("<I", index))
        vocabulary_hash.update(gram.encode("utf-8") + b"\0")
    rows = 1 if nclasses == 2 else nclasses
    idf = floats(reader, width, "d")
    biases = floats(reader, rows, "d")
    weights = floats(reader, rows * width, "f")
    if reader.pos != len(body):
        raise ValueError("unexpected CCM2 trailing bytes")
    return {
        "file_sha256": sha256(raw).hexdigest(),
        "bytes": len(raw),
        "width": width,
        "classes": classes,
        "ngram_range": [gram_min, gram_max],
        "flags": flags,
        "vocabulary_sha256": vocabulary_hash.hexdigest(),
        "idf": idf,
        "bias": biases,
        "weights": weights,
    }


def tensor_differences(left, right):
    if len(left) != len(right):
        raise ValueError("cannot compare unaligned tensors")
    count = exact = above_1e6 = above_1e4 = above_1e2 = sign_flips = 0
    total_abs = total_sq = max_abs = 0.0
    for a, b in zip(left, right):
        count += 1
        delta = abs(a-b)
        exact += int(a == b)
        above_1e6 += int(delta > 1e-6)
        above_1e4 += int(delta > 1e-4)
        above_1e2 += int(delta > 1e-2)
        sign_flips += int((a < 0 < b) or (b < 0 < a))
        total_abs += delta
        total_sq += delta * delta
        max_abs = max(max_abs, delta)
    return {
        "values": count,
        "bitwise_equal_values": exact,
        "changed_values": count-exact,
        "abs_delta_gt_1e-6": above_1e6,
        "abs_delta_gt_1e-4": above_1e4,
        "abs_delta_gt_1e-2": above_1e2,
        "sign_flips": sign_flips,
        "max_abs_delta": max_abs,
        "mean_abs_delta": total_abs / count if count else 0.0,
        "rms_delta": sqrt(total_sq / count) if count else 0.0,
    }


def compare(reference: Path, candidate: Path) -> dict:
    a = load_ccm2(reference)
    b = load_ccm2(candidate)
    aligned = (
        a["width"] == b["width"]
        and a["classes"] == b["classes"]
        and a["vocabulary_sha256"] == b["vocabulary_sha256"]
        and a["ngram_range"] == b["ngram_range"]
        and a["flags"] == b["flags"]
    )
    result = {
        "schema": "codeccat-ccm2-numerical-difference-v1",
        "reference_sha256": a["file_sha256"],
        "candidate_sha256": b["file_sha256"],
        "identical_files": a["file_sha256"] == b["file_sha256"],
        "same_class_order": a["classes"] == b["classes"],
        "same_vocabulary": a["vocabulary_sha256"] == b["vocabulary_sha256"],
        "same_feature_profile": (
            a["ngram_range"] == b["ngram_range"]
            and a["flags"] == b["flags"]
        ),
        "reference_features": a["width"],
        "candidate_features": b["width"],
        "reference_classes": len(a["classes"]),
        "candidate_classes": len(b["classes"]),
        "reference_bytes": a["bytes"],
        "candidate_bytes": b["bytes"],
        "aligned_for_numeric_comparison": aligned,
    }
    if aligned:
        result["idf"] = tensor_differences(a["idf"], b["idf"])
        result["intercept"] = tensor_differences(a["bias"], b["bias"])
        result["weights"] = tensor_differences(a["weights"], b["weights"])
        width = a["width"]
        per_class = []
        for idx, label in enumerate(a["classes"]):
            start = idx * width
            delta = tensor_differences(
                a["weights"][start:start+width],
                b["weights"][start:start+width],
            )
            per_class.append({
                "encoding": label,
                "max_abs_delta": delta["max_abs_delta"],
                "mean_abs_delta": delta["mean_abs_delta"],
                "changed_values": delta["changed_values"],
            })
        result["top_classes_by_weight_delta"] = sorted(
            per_class, key=lambda r: (-r["mean_abs_delta"], r["encoding"])
        )[:10]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MODEL_NUMERIC_COMPARISON", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
