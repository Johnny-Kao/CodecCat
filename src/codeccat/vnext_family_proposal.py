"""Research-only family proposal from independent decoded-text evidence.

No training-label lookups, no model weight changes, no production integration.
Scans learned-codec candidates beyond the existing Top3; expensive by design.
"""
from __future__ import annotations
import codecs
import math
import unicodedata
from dataclasses import dataclass
import numpy as np

SCRIPTS = ("LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "HIRAGANA",
           "KATAKANA", "HANGUL", "THAI", "DEVANAGARI", "ARMENIAN", "GEORGIAN")

@dataclass(frozen=True)
class Proposal:
    encoding: str | None
    changed: bool
    examined: int
    reason: str
    language_gain: float | None

def decode_prefix(raw: bytes, codec: str, *, final: bool) -> str | None:
    try:
        dec = codecs.getincrementaldecoder(codec)("strict")
        return dec.decode(raw[:4096], final=final and len(raw)<=4096)
    except (UnicodeError, LookupError, ValueError):
        return None

def script_coherence(text: str) -> float | None:
    """Character-script purity among Unicode letters; common digits ignored."""
    counts = {}
    for ch in text[:4096]:
        if not unicodedata.category(ch).startswith("L"):
            continue
        name = unicodedata.name(ch, "")
        script = next((s for s in SCRIPTS if name.startswith(s+" ")), None)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
    total = sum(counts.values())
    if total < 16:
        return None
    return max(counts.values()) / total

def propose(model, language, raw: bytes, ranked, *, max_candidates: int = 32,
            language_advantage: float = 1.0, min_script_delta: float = 0.0) -> Proposal:
    if not isinstance(raw, bytes):
        raise TypeError("expected bytes")
    base = ranked.encoding
    if base is None or len(ranked.candidates) <= 1:
        return Proposal(base, False, 0, "no-ambiguity", None)
    if ranked.bom_hint:
        return Proposal(base, False, 0, "bom-decisive", None)
    original_text = decode_prefix(raw, base, final=len(raw)<4096)
    if original_text is None:
        return Proposal(base, False, 0, "invalid-baseline", None)
    original_lang = language.score(original_text)
    original_script = script_coherence(original_text)
    if original_lang is None:
        return Proposal(base, False, 0, "short-baseline", None)
    row = model.vectorizer.transform([raw[:4096].decode("latin1")]).getrow(0)
    margins = np.asarray(model.classifier.coef_[:, row.indices] @ row.data + model.classifier.intercept_)
    classes = list(model.classifier.classes_)
    if len(classes) == 2:
        margins = np.array([-float(margins[0]), float(margins[0])])
    order = np.argsort(margins)[::-1]
    count = 0
    best_name = base
    best_gain = 0.
    for ix in order:
        name = str(classes[int(ix)])
        decoded = decode_prefix(raw, name, final=len(raw)<4096)
        if decoded is None:
            continue
        count += 1
        if count > max_candidates:
            break
        if name == base or decoded == original_text:
            continue
        alternative = language.score(decoded)
        if alternative is None:
            continue
        purity = script_coherence(decoded)
        script_gain = (purity - original_script if purity is not None and original_script is not None else 0.)
        if script_gain < min_script_delta:
            continue
        gain = alternative - original_lang
        if gain > best_gain:
            best_gain, best_name = gain, name
    if best_name != base and best_gain > language_advantage:
        return Proposal(best_name, True, count, "independent-language-family", best_gain)
    return Proposal(base, False, count, "no-strong-family-evidence", best_gain)
