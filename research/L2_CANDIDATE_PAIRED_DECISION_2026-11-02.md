# L2 candidate paired decision — 2026-11-02

GitHub Actions: https://github.com/Johnny-Kao/CodecCat/actions/runs/41357810701 (PASS).

Compared Frozen Stable23 baseline vs candidate 1–2 gram using 3,086 grouped-fold chardet/test-data fixtures, and eight non-identical charset-normalizer examples. Both evaluated under identical folds within one CI job.

| Metric | Stable23 | 1–2 gram |
|---|---:|---:|
| Correct / 3086 | 2830 | 2868 |
| Accuracy | 91.70% | 92.94% |
| Exclusive correct | 39 | 77 |
| External / 8 | 7 | 7 |
| 3-fold training process CPU s | 18.925 | 9.699 |
| Median across fold p50 inference CPU ms | 2.4761 | 1.8154 |
| Max fold p95 CPU ms | 4.3772 | 3.0724 |
| Process peak RSS MiB | 294.31 | 294.31 |

Family regression audit: 9 families decrease, 33 increase, 25 total family losses and 63 gains. Severe declines: cp037 30→24 (-6); cp850 38→32 (-6); cp437 30→25 (-5); cp858 24→21 (-3). Gains include utf-7 (+7), hp-roman8 (+6), mac-roman (+5), cp500 (+4), koi8-u (+4). UTF-8 loses one case (252→251). These are source-specific development results, not externally certified gains.

**Decision: NOT READY to replace bundled default.** Candidate has positive overall and speed results, but category regressions and 8-case external diagnostic do not meet release confidence. Maintain existing default. Do not introduce hand-tuned per-encoding overrides without independent validation. Results archived as downloadable CI artifact named l2-candidate-full-paired-audit. Research branch only, not main/Alpha/PyPI.
