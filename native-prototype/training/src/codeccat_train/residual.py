"""Stable23 residual-sample selection for offline replay.

This is the *data-selection* stage used by the historical bundled-model
recipe. It is separate from the fitting algorithm and does not download data.
"""
from __future__ import annotations
from collections import Counter

_BOMS = (
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xfe\xff", "utf-16-be"),
    (b"\xff\xfe", "utf-16-le"),
)


def stable_structural_guess(data: bytes, known: set[str]) -> str | None:
    # Match the former Stable23 selection rule, not the full Runtime gate.
    final = len(data) < 4096
    if not final and data and any(
        sig.startswith(data) and len(data) < len(sig) for sig, _ in _BOMS
    ):
        return None
    for signature, encoding in _BOMS:
        if data.startswith(signature):
            family = "-".join(encoding.split("-")[:2])
            preferred = (
                family if family in ("utf-16", "utf-32") and family in known
                else encoding
            )
            return preferred if preferred in known else None
    if len(data) < 32:
        return None
    pairs = min(len(data), 4096) // 2
    even = data[: pairs * 2 : 2].count(0)
    odd = data[1 : pairs * 2 : 2].count(0)
    if max(even, odd) >= max(12, int(pairs * 0.30)) and max(
        even, odd
    ) >= 4 * max(1, min(even, odd)):
        predicted = "utf-16-be" if even > odd else "utf-16-le"
        if predicted in known:
            return predicted
    return None


def residual_subset(rows, min_anchors: int = 2, max_drop_fraction: float = 0.35):
    """Return selected input rows and per-class drop counts, preserving order.

    A row is any sequence starting with (bytes, encoding, group); remaining
    metadata is preserved. Filtering applies to training samples only.
    """
    labels = {row[1] for row in rows}
    per_label = Counter(row[1] for row in rows)
    drop_limit = {
        label: min(per_label[label] - min_anchors,
                   int(per_label[label] * max_drop_fraction))
        for label in per_label
    }
    covered = Counter()
    dropped = Counter()
    chosen = []
    for row in rows:
        data, label = row[:2]
        if stable_structural_guess(data, labels) == label and covered[label] < drop_limit[label]:
            covered[label] += 1
            dropped[label] += 1
        else:
            chosen.append(row)
    if {row[1] for row in chosen} != labels:
        raise ValueError("residual selection removed a training codec class")
    return chosen, dict(dropped)
