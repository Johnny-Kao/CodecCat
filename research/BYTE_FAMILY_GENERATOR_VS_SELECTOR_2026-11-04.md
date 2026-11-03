# Fast byte-family layer: candidate coverage PASS, final selection FAIL — 2026-11-04

## Code and reproducibility
- Research-only `src/codeccat/vnext_byte_families.py`; histograms over 4096-byte inference prefix, 256 bins, class-wise reference centroids learned on training source, no change to base classifier weights.
- Coverage CI https://github.com/Johnny-Kao/CodecCat/actions/runs/41664177485 — PASS
- Paired fixed-gap decision CI https://github.com/Johnny-Kao/CodecCat/actions/runs/41664628576 — PASS
- Three grouped folds with Binary TF SGD alpha 0.0003 fixed; Enca corpus commit e6a62fbfa049c2c078963a042da89310a6a96e6e, full/prefix SHA overlaps excluded.

## Candidate coverage (not final accuracy)
| | Original top3 | +3 family | +8 family | +16 family |
|---|---:|---:|---:|---:|
| Development 3086 text-equivalent true decoding reachable | 3046 | 3057 | 3067 | 3075 |
| Enca 81 | 53 | 57 | 60 | 70 |

Cheap vector-scoring alone median CPU p50 ≈0.05 ms (development) / 0.049 ms (Enca), excluding codec validation and primary model rank.
- +16 family candidates can make 17 additional Enca cases reachable but does not select them correctly without an independent reliable decision stage.

## Final prediction gate
| score gap | dev correct /3086 | fixed | broken | Enca correct /81 |
|---|---:|---:|---:|---:|
| baseline | 2897 | 0 | 0 | 45 |
| 0.05 | 2515 | 20 | 402 | 34 |
| 0.10 | 2600 | 14 | 311 | 35 |
| 0.20 | 2718 | 8 | 187 | 43 |
| 0.35 | 2862 | 3 | 38 | 44 |

At gap 0.10 median extra CPU ≈0.039 ms in development (includes scanning shortlist, similarity calculation, strict validation on selected codec; excludes initial rank), but 311 new regressions dominate.

## Verdict
- Family centroid generator: **research candidate coverage PASS**, no correctness promotion, may be reused after separate selectors prove generalization.
- Centroid gap winner override: **FAIL** for all tested margins. Do not tune the margin to external Enca fixtures; no release integration.
- Main blocker now is candidate selection/calibration with independent differentiating evidence, not CPU, not raw candidate availability alone. The Enca source comprises correlated texts across encodings and does not prove population error rates.
- Current shipped `v0.1.0a2` unchanged.
