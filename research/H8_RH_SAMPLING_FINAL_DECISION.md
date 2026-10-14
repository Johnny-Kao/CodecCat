# H8 RH Uniform Sampling — Rejected

Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/38576370218

## Decision
Reject fixed evenly spaced RH-768 byte sampling with frozen R21 weights. Keep R21 unchanged.

| Cohort | N | Frozen R21 | Uniform RH candidate | chardet 7 | charset-normalizer |
|---|---:|---:|---:|---:|---:|
| Exploratory (2025-13/38/51, 2026-04) | 1,026 | 925 (90.16%) | 877 (85.48%) | 935 (91.13%) | 947 (92.30%) |
| Newly evaluated (2024-22/46/51) | 774 | 704 (90.96%) | 683 (88.24%) | 690 (89.15%) | 733 (94.70%) |
| Total | 1,800 | 1,629 (90.50%) | 1,560 (86.67%) | 1,625 (90.28%) | 1,680 (93.33%) |

Uniform candidate changed 182 predictions, beneficial 44, harmful 113. Median inference time increased from 142,620 ns/sample to 179,739 (+26%). Failed accuracy and speed criteria.

The representation was changed without retraining weights; this demonstrates the frozen model is not invariant to the new sampling distribution. It does *not* prove evenly spaced samples are intrinsically inferior when trained correctly. Do not tune sample size or spacing against these crawls.

## Product bottleneck
Final random 2025/2026 benchmark n=1,026: R21 925 correct vs chardet 935 and charset-normalizer 947. RH n=574: R21 513 vs chardet 543; true CJK n=335: R21 303 vs chardet 320. Both are marginal summaries; cross-tab overlap is not established.

## Memory baseline
Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/38575796406
Independent Python processes, Ubuntu, static synthetic byte inputs:
- CodecCat RSS after init 44,084 KiB; 64 KiB post-inference 44,988 KiB
- chardet7 RSS after init 22,316 KiB; 64 KiB post-inference 22,444 KiB
- charset-normalizer RSS after init 21,976 KiB; 64 KiB post-inference 28,808 KiB

All are OS process RSS, not model-only allocated bytes. CodecCat runtime depends on NumPy; use isolated import-vs-model attribution before any memory optimization. Do not risk accuracy by weight conversion or runtime simplification without independent validation.

## Next architecture decision
Avoid further sampling/threshold sweeps. The next promising general approach is RH encoding-family evidence that preserves frozen scorer decisions where evidence is inconclusive, but evaluate candidate coverage and label semantics first. Any new learned representation needs a separate new candidate trained on development data; do not alter R21. All known 2024-2026 crawls used in H1-H8 are development/evaluation-exposed; do not call them untouched holdout again.
