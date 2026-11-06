"""Offline L3 unicode-trigram fitting for optional model research.

Matches historical LanguageCoherence.fit behavior without importing CodecCat
Runtime, legacy training code, or third-party corpus into this distribution.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "vnext-l3-unicode-trigram-v1"
_SPACES = re.compile(r"\s+")


def _trigrams(text: str) -> Counter[str]:
    text = _SPACES.sub(" ", text[:4096].casefold())
    return Counter(text[i:i + 3] for i in range(max(0, len(text)-2)))


def _decode(data: bytes, encoding: str) -> str | None:
    try:
        decoder = codecs.getincrementaldecoder(encoding)("strict")
        return decoder.decode(data[:4096], final=len(data) < 4096)
    except (UnicodeError, LookupError, ValueError):
        return None


def fit_l3(training_rows, max_features: int = 30000) -> dict:
    if max_features < 100:
        raise ValueError("max_features must be >= 100")
    counts = Counter()
    fingerprint = hashlib.sha256()
    found = 0
    for row in training_rows:
        raw, codec, group = row[:3]
        if not isinstance(raw, bytes) or not isinstance(codec, str) or not isinstance(group, str) or not codec or not group:
            raise ValueError("expected (bytes, encoding, source_group)")
        # Historical corpus() passed a truncated prefix to L3 training.
        # Reproduce that stage when source files themselves exceed 4096 B.
        raw = raw[:4096]
        text = _decode(raw, codec)
        if text is None:
            continue
        counts.update(_trigrams(text))
        fingerprint.update(hashlib.sha256(raw).digest())
        fingerprint.update(codec.encode() + b"\x00" + group.encode() + b"\x00")
        found += 1
    if not found or not counts:
        raise ValueError("no decodable training text")
    grams = dict(sorted(counts.most_common(max_features), key=lambda x: x[0]))
    return {
        "schema": SCHEMA,
        "grams": grams,
        "total": sum(grams.values()),
        "fingerprint": fingerprint.hexdigest(),
        "alpha": 0.5,
    }


def save_l3(training_rows, destination: Path, max_features: int = 30000) -> dict:
    fitted = fit_l3(training_rows, max_features=max_features)
    # Historical LanguageCoherence.save used json.dumps with these options
    # and wrote no trailing newline.
    payload = json.dumps(fitted, sort_keys=True, ensure_ascii=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    return {
        "schema": SCHEMA,
        "model_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "model_bytes": len(payload.encode("utf-8")),
        "training_fingerprint": fitted["fingerprint"],
        "training_grams": len(fitted["grams"]),
        "training_total_trigram_count": fitted["total"],
        "max_features": max_features,
        "optional_model_enabled_by_default": False,
    }
