# One-shot L2 matrix — 2026-11-02

Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41356761841 — eight of eight matrix jobs PASS.

All variants trained on identical grouped folds of 3,086 source-fixture texts; independent-source diagnostic consists of **only eight deduplicated charset-normalizer test documents**, insufficient for production promotion. No model or package change.

| Variant | Correct / 3086 | Difference | Ext correct / 8 | Train CPU sec (3 folds) | p50 CPU ms* | p95 CPU ms* | peak process RSS MiB* |
|---|---:|---:|---:|---:|---:|---:|---:|
| residual Stable23 baseline | 2828 | — | 7 | 15.98 | 1.996 | 3.729 | 291.25 |
| residual 1–2 gram | **2868** | **+40** | 7 | 9.77 | 1.832 | 3.138 | 224.79 |
| residual 1–3 gram | 2837 | +9 | 7 | 20.07 | 3.132 | 5.741 | 339.21 |
| full training 2–3 gram | 2800 | −28 | 7 | 22.90 | 2.998 | 5.861 | 288.48 |
| residual 2–4 gram | 2785 | −43 | 7 | 32.62 | 3.732 | 7.155 | 537.41 |
| strong regularization | 2765 | −63 | 7 | 11.22 | 1.501 | 2.907 | 290.76 |
| LinearSVC | 2751 | −77 | 7 | 22.36 | 2.443 | 4.454 | 303.61 |
| RidgeClassifier | 2750 | −78 | 7 | 37.35 | 2.449 | 4.452 | 314.46 |

*Metrics measured per separate CI job, may include model train/import (RSS) and environment fluctuations. Comparison is directional, not definitive isolated inference memory.

Inference top-three failure count improved for 1–2 grams: top-three missing correct text 80 vs 105 baseline; remaining wrong 218 vs 258.

Interpretation: 1–2 gram is the only promising exploratory candidate with simultaneous development accuracy and resource improvements. Different model characteristics and near-duplicate semantics not audited. The eight external cases show no gain but no loss (7/8). Testing multiple variants on the same development folds creates a selection bias: a locked external validation is still required, and per-family regressions must be compared before adoption.

**Decision: retain current stable23 default.** Next batch should compare candidate and baseline on disjoint independent data, per-family paired losses, clean inference-only CPU/RAM and deterministic model serialization. No automatic promotion.
