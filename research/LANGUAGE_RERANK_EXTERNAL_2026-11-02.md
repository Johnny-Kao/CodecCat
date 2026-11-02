# Fixed Unicode reranker — external-source validation (2026-11-02)

## Exact evidence
- GitHub Actions: https://github.com/Johnny-Kao/CodecCat/actions/runs/41463887180 (PASS).
- Source: charset-normalizer fixture files supplied by its repository; 18 listed candidates. Hash and 4096-byte inference-prefix overlap with chardet test-data training corpus removed 10, leaving 8 eligible files.
- Fixed research model: Binary TF 1–2 gram, SGD alpha 0.0003, trained on full chardet corpus after residual exclusion.
- Fixed reranker (chosen solely in earlier development experiment): max_l2_margin 0.5, min_language_advantage 0.15. Language trigrams learned from training corpus only.

## Results
- Without language: 7/8 text-equivalence correct.
- With language: 7/8 text-equivalence correct; 0 changes, 0 fixed, 0 broken.
- One error is cp1252 labeled text predicted iso8859-1; reranker declined to run because L2 margin exceeded gate.
- Base CPU p50 across eight inputs 1.935668 ms. Additional rerank call CPU p50 0.006982 ms; most were no-op gates. This is **not** the latency cost in eligible, language-scored cases.
- Peak process RSS 225.44 MiB includes Python sklearn, training both models; cannot represent inference-only runtime footprint.

## Decision
Do not promote the development-selected language thresholds. External support is insufficient: N=8, no changed predictions, filename labels not independently audited. Do not tune thresholds to fix the one cp1252 case. More genuinely unseen source-diverse fixtures and calibrated language-evidence costs are required. Existing Alpha remains unmodified.
