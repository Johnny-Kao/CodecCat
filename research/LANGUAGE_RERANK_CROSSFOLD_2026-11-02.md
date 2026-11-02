# Cross-fold language coherence reranking — 2026-11-02

CI: https://github.com/Johnny-Kao/CodecCat/actions/runs/41460289813 — PASS.

Fixed base weights Binary TF 1–2 gram, SGD modified_huber alpha=0.0003. Unicode trigram LanguageCoherence fitted only on each training fold; grouped 3-fold evaluation on 3086 chardet/test-data fixtures.

| max_l2_margin | min_language_advantage | correct | fixed | broken | changed |
|---:|---:|---:|---:|---:|---:|
| base | N/A | 2897 | 0 | 0 | 0 |
| 0.2 | 0.15 | 2907 | 14 | 4 | 21 |
| **0.5** | **0.15** | **2912** | **19** | **4** | **26** |
| 1.0 | 0.15 | 2905 | 22 | 14 | 39 |
| 0.5 | 0.4 | 2905 | 9 | 1 | 12 |
| 1.0 | 0.4 | 2899 | 9 | 7 | 18 |
| 2.0 | 0.4 | 2898 | 9 | 8 | 19 |

The chosen 0.5/0.15 improves the development corpus from 93.88% to 94.36%, a net +15 correct cases, but threshold selection on these same held-out folds adds development-set selection bias. It is **not** evidence that the secondary layer generalizes to independent sources. Score margins are uncalibrated and depend on classifier settings.

## Decision
Keep existing packaged Stable23 and the experimental Binary TF unchanged; no runtime feature enabled by default. Next validate the chosen fixed thresholds against previously unused source-diverse documents, as well as per-encoding regression and latency/RSS; do not tune on those holdouts. The result supports exploring independent language evidence rather than only L2 parameter tuning.
