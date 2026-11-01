# Selective L3 external and lifecycle preflight — 2026-11-01

**GitHub Actions SUCCESS:** https://github.com/Johnny-Kao/CodecCat/actions/runs/41310887091

## Results

Evaluation of 21 nonidentical-normalized-text third-party uchardet fixtures, against a stable23 residual model trained on pinned chardet/test-data.

| Metric | Result |
|---|---:|
| L2 baseline correct decoded text | 20/21 |
| Selective L3 correct decoded text | **19/21** |
| L3 calls | 3 |
| L3 decision changes | 1 |
| Fixes | 0 |
| Harms | **1** |
| L3 save/load parity | 21/21 |
| L3 JSON model bytes | 376,410 |
| L2 rank median measured time | 619,582 ns |
| L3 marginal median time | 2,604 ns |
| L3 marginal p95 | 64,046 ns |
| Whole experiment peak RSS | 320.5 MiB |

This **contradicts** the development-only improvement on 3,086 in-source samples (+10 fixes, 0 harms). Thus **NO promotion**. The 21 samples are not a new independently verified natural-source corpus, but a single regression is already disqualifying for unconditional default-on.

Timing samples are heterogeneous and only 3/21 trigger L3; do not interpret median 2.6 us as invoked-L3 latency. RSS includes Python, sklearn dependencies and training: not inference-only RAM.

## Decision

- Keep L3 disabled by default in released Alpha and main.
- Keep existing retrainable L3 for optional user experimentation, not unconditional production reranking.
- Investigate the one harmful case from the Actions `cases.jsonl`, with evidence and without a case-specific exception rule.
- Test any robust uncertainty/verification design by source-separated group evaluation with fixed predeclared gates and independent original documents, not posthoc margin tuning.
- No changes to production model, alpha, main, PyPI.

Artifact: `selective-l3-preflight` includes `summary.json` and full `cases.jsonl`.
