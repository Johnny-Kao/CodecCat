"""Optional L3: Unicode text-coherence evidence, learned from training texts only.

This verifier never learns codec labels: it models decoded Unicode trigrams.
It can change L2's top-two decision only when both codecs strictly decode,
the byte-level margin is small, and learned textual evidence is strong.
Disabled unless the caller explicitly invokes rerank().
"""
from __future__ import annotations

import codecs
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCHEMA = "vnext-l3-unicode-trigram-v1"
_SPACE = re.compile(r"\\s+")


def _normalized(text: str) -> str:
    return _SPACE.sub(" ", text.casefold())


def _trigrams(text: str) -> Counter[str]:
    text = _normalized(text[:4096])
    return Counter(text[i:i + 3] for i in range(max(0, len(text) - 2)))


def _decode(data: bytes, encoding: str, *, final: bool) -> str | None:
    try:
        decoder = codecs.getincrementaldecoder(encoding)("strict")
        return decoder.decode(data[:4096], final=final and len(data) <= 4096)
    except (UnicodeError, LookupError, ValueError):
        return None


@dataclass(frozen=True)
class Rerank:
    encoding: str | None
    called: bool
    changed: bool
    reason: str
    language_advantage: float | None = None


@dataclass(frozen=True)
class LanguageCoherence:
    grams: dict[str, int]
    total: int
    fingerprint: str
    alpha: float = 0.5
    schema: str = SCHEMA

    @classmethod
    def fit(
        cls,
        samples: Iterable[tuple[bytes, str, str]],
        *,
        max_features: int = 30000,
    ) -> "LanguageCoherence":
        if max_features < 100:
            raise ValueError("max_features must be >= 100")
        counts: Counter[str] = Counter()
        fingerprint = hashlib.sha256()
        found = 0
        for raw, codec, group in samples:
            if not isinstance(raw, bytes) or not isinstance(codec, str) or not isinstance(group, str) or not codec or not group:
                raise ValueError("expected (bytes, encoding, source_group)")
            text = _decode(raw, codec, final=len(raw) < 4096)
            if text is None:
                continue
            counts.update(_trigrams(text))
            fingerprint.update(hashlib.sha256(raw).digest())
            fingerprint.update(codec.encode() + b"\\x00" + group.encode() + b"\\x00")
            found += 1
        if not found or not counts:
            raise ValueError("no decodable training text")
        grams = dict(sorted(counts.most_common(max_features), key=lambda x: x[0]))
        return cls(grams, sum(grams.values()), fingerprint.hexdigest())

    def score(self, text: str) -> float | None:
        grams = _trigrams(text)
        if sum(grams.values()) < 12:
            return None
        base = self.total + self.alpha * (len(self.grams) + 1)
        logbase = math.log(base)
        value = sum(
            count * (math.log(self.grams.get(trigram, 0) + self.alpha) - logbase)
            for trigram, count in grams.items()
        )
        return value / sum(grams.values())

    def rerank(
        self,
        ranked,
        data: bytes,
        *,
        final: bool = True,
        max_l2_margin: float = 0.20,
        min_language_advantage: float = 0.15,
    ) -> Rerank:
        """Do not use on decisive L1 outcomes or on uncalibrated singletons."""
        baseline = ranked.encoding
        if not isinstance(data, bytes):
            raise TypeError("expected bytes")
        if baseline is None or len(ranked.candidates) < 2 or len(ranked.scores) < 2:
            return Rerank(baseline, False, False, "no-competing-candidate")
        if ranked.candidates[0] != baseline:
            raise ValueError("ranked top candidate disagrees with encoding")
        margin = float(ranked.scores[0]) - float(ranked.scores[1])
        if not math.isfinite(margin) or margin < 0 or margin > max_l2_margin:
            return Rerank(baseline, False, False, "l2-margin-not-eligible")
        first, second = ranked.candidates[:2]
        original = _decode(data, first, final=final)
        alternative = _decode(data, second, final=final)
        if original is None or alternative is None:
            return Rerank(baseline, True, False, "invalid-decoded-candidate")
        if original == alternative:
            return Rerank(baseline, False, False, "identical-decoded-text")
        a, b = self.score(original), self.score(alternative)
        if a is None or b is None:
            return Rerank(baseline, True, False, "insufficient-language-evidence")
        advantage = b - a
        if advantage > min_language_advantage:
            return Rerank(second, True, True, "language-evidence", advantage)
        return Rerank(baseline, True, False, "no-strong-language-advantage", advantage)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "schema": self.schema, "grams": self.grams,
            "total": self.total, "fingerprint": self.fingerprint,
            "alpha": self.alpha,
        }, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LanguageCoherence":
        path = Path(path)
        if path.stat().st_size > 8_000_000:
            raise ValueError("language model too large")
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("schema") != SCHEMA:
            raise ValueError("language model schema mismatch")
        grams = obj.get("grams")
        if (not isinstance(grams, dict) or not grams or len(grams) > 30000
                or any(not isinstance(k, str) or len(k) != 3 or not isinstance(v, int) or v <= 0 for k, v in grams.items())):
            raise ValueError("invalid language vocabulary")
        total = obj.get("total")
        alpha = obj.get("alpha")
        if total != sum(grams.values()) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("invalid language model statistics")
        fp = obj.get("fingerprint")
        if not isinstance(fp, str) or len(fp) != 64:
            raise ValueError("invalid training fingerprint")
        return cls(grams, total, fp, float(alpha))
