# Accuracy improvement batch, 2026-11-02

## Evidence
- Seven-configuration grouped-fold batch: https://github.com/Johnny-Kao/CodecCat/actions/runs/41374940076 (7/7 PASS).
- Paired comparison with stable23: https://github.com/Johnny-Kao/CodecCat/actions/runs/41375046706 (PASS).
- Fixed chardet/test-data corpus: 3,086 documents, grouped 3-fold. Source set is competitor-maintained; not a blind real-world benchmark.
- Same selected charset-normalizer independent-byte holdout: 8 documents, all candidate variants 7/8.

## Tested variants, correct of 3,086
| Variant | Correct | Accuracy |
|---|---:|---:|
| Stable23 default | 2,830 in paired run | 91.70% |
| Binary 1–2 gram original | 2,882 | 93.39% |
| Binary 1–2 gram, full train | 2,892 | 93.71% |
| Binary 1–2 gram, class balanced | 2,866 | 92.87% |
| Binary 1–2 gram, min_df=1 | 2,882 | 93.39% |
| Binary 1–2 gram, alpha 0.00003 | 2,829 | 91.67% |
| **Binary 1–2 gram, alpha 0.0003** | **2,897** | **93.88%** |

## Paired decision
- Baseline 2,830 vs winning candidate 2,897: +67 correct, +2.17 percentage points, development corpus.
- Candidate-only correct 96; baseline-only correct 29.
- Improved encoding categories 35, regressed 8. Some categories are very small, and CP864 fell 2/2 -> 0/2; CP437 -2, KZ1048 -2, PTCP154 -2.
- External mini-holdout remains 7/8 vs 7/8; not enough evidence for production adoption.
- Median fold inference CPU 2.58ms baseline vs 1.86ms candidate. Separate process RSS not measured in this pair.
- Estimated chardet7.5 score on the related 3094-case previous mixed benchmark is 3013/3094 (97.38%), not strictly identical samples; **do not claim exact matched delta** without per-case paired rerun.

## Conclusion
Best current Python L2 research candidate: 1–2gram Binary TF with SGD modified_huber alpha=0.0003, trained on residual subset. Reached 93.88% on development fixtures; misses 95% target by 35 more correct predictions (2932 required). Avoid cherry-picking on repeatedly reused development samples. Keep shipped v0.1.0a2 Stable23 unchanged. Next useful experiment is not more arbitrary hyperparameter tuning; it is independent-source validation and source-generalizable feature improvements targeting the still-missed labels.
