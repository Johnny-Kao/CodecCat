"""Audit a release wheel for raw corpus files and readable model structure.

This is a distribution-content audit, not a legal clearance or memorization test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path

ALLOWED_MODEL_FILES = {
    "codeccat/models/default.ccm",
    "codeccat/models/default.ccm.meta.json",
    "codeccat/models/language-l3.json",
}
RAW_SUFFIXES = (".txt", ".html", ".htm", ".xml", ".csv", ".pdf", ".md", ".rst")
# Sample basenames cannot be included as package artifacts.
SUSPICIOUS_PREFIXES = ("culturax_", "cc_", "_ude_", "_enca_", "_uchardet_", "archive_")


def parse_model(data: bytes) -> dict:
    if len(data) < 48 or data[:4] not in (b"CCM1", b"CCM2"):
        raise ValueError("unrecognized native model")
    body, digest = data[:-32], data[-32:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("bad CCM content digest")
    schema, width, classes, gram_min, gram_max, flags, reserved = struct.unpack_from(
        "<4sIIBBBB", body
    )
    if not (0 < width <= 200_000 and 2 <= classes <= 256 and
            1 <= gram_min <= gram_max <= 3 and reserved == 0):
        raise ValueError("unexpected model dimensions")
    pos = 16

    def read_string():
        nonlocal pos
        if pos + 2 > len(body):
            raise ValueError("truncated string length")
        size = struct.unpack_from("<H", body, pos)[0]
        pos += 2
        if size > 256 or pos + size > len(body) or size == 0:
            raise ValueError("invalid model string length")
        value = body[pos:pos + size].decode("utf-8")
        pos += size
        return value

    labels = [read_string() for _ in range(classes)]
    longest_gram = 0
    indexes = set()
    for _ in range(width):
        if pos + 4 > len(body):
            raise ValueError("truncated vocabulary")
        idx = struct.unpack_from("<I", body, pos)[0]
        pos += 4
        gram = read_string()
        if idx >= width or idx in indexes or not gram_min <= len(gram) <= gram_max:
            raise ValueError("unexpected model vocabulary")
        indexes.add(idx)
        longest_gram = max(longest_gram, len(gram))
    rows = 1 if classes == 2 else classes
    expected_tail = (width + rows) * 8 + width * rows * (4 if schema == b"CCM2" else 8)
    if pos + expected_tail != len(body):
        raise ValueError("unexpected model payload size")
    return {
        "schema": schema.decode(),
        "features": width,
        "classes": len(labels),
        "max_vocab_characters": longest_gram,
        "model_sha256": hashlib.sha256(data).hexdigest(),
    }


def audit(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        codeccat = [n for n in names if n.startswith("codeccat/")]
        if len(codeccat) != len(set(codeccat)):
            raise ValueError("duplicate wheel paths")
        for name in codeccat:
            basename = name.rsplit("/", 1)[-1].lower()
            if basename.startswith(SUSPICIOUS_PREFIXES):
                raise ValueError("unexpected corpus fixture: " + name)
            if name.startswith("codeccat/models/") and name not in ALLOWED_MODEL_FILES:
                raise ValueError("unexpected bundled model file: " + name)
            if name.startswith("codeccat/models/") and name.lower().endswith(RAW_SUFFIXES):
                raise ValueError("raw text artifact: " + name)
        if "codeccat/models/default.ccm" not in names:
            raise ValueError("default model missing")
        stats = parse_model(z.read("codeccat/models/default.ccm"))
        sidecar = json.loads(z.read("codeccat/models/default.ccm.meta.json"))
        if sidecar.get("model_sha256") != stats["model_sha256"]:
            raise ValueError("provenance sidecar does not match model")
        language = json.loads(z.read("codeccat/models/language-l3.json"))
        grams = language.get("grams")
        if not isinstance(grams, dict) or not grams or any(
            not isinstance(k, str) or len(k) != 3 or
            not isinstance(v, int) or v <= 0 for k, v in grams.items()
        ):
            raise ValueError("unexpected language model content")
        return {
            "wheel": path.name,
            "model": stats,
            "l3_trigrams": len(grams),
            "max_l3_characters": 3,
            "raw_training_files_bundled": False,
            "scope": "artifact-structure-only",
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("wheel", type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.wheel), sort_keys=True, indent=2))
