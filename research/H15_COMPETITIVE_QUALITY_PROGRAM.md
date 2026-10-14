# CodecCat H15 — Competitive Quality Program

## Release target
Beat BOTH chardet and charset-normalizer on independently verified, previously untouched bytes, not merely be faster. Frozen R21 remains unchanged. Do not edit main, publish releases or PyPI.

## Literature and code references
- https://chardet.readthedocs.io/en/stable/how-it-works.html — layered deterministic validity, CJK structural probing, and bigram statistics.
- https://chardet.readthedocs.io/en/6.0.0/how-it-works.html — Japanese hiragana-context model.
- https://charset-normalizer.readthedocs.io/en/stable/api.html — decoded character coherence and mess ratios.

## Evidence
H11: 19 of 28 RH/CJK errors miss correct label in raw Top-3; 17 are Japanese encodings. H14: +41/3355 correct on exposed crawls at +19% inference time, but the data are not a new holdout.

## One bounded wide tournament, no threshold-by-threshold experiments
Batch evaluate on identical cached bytes (same run):
0. Frozen R21.
1. H14 reference.
2. Strict SJIS + early cheap eligibility filters.
3. Strict SJIS + EUC-JP dual evidence and conflict abstention.
4. Encoding structural-state machine evidence.
5. Japanese kana context transition signals.
6. Ordered byte pair features, not modulo-summed pairs.
7. Selective family evidence reranking.
8. Decode-coherence/controls evidence, used as features, not unconditional overrides.
Candidates 5 and 6 require separate development-only training; no model weights are to be refitted on held-out data.

## Corpus integrity
- Source new 2023-or-earlier crawls and independent known-encoding fixtures; previously used 2024-2026 crawls are development data only.
- Exact-byte hash deduplication, grouping by crawl, WARC and domain; freeze seed, samples and acceptance metric before candidate results.
- Header declarations do NOT guarantee actual encoding. Preserve label provenance; build a verified subset of known-source bytes and synthetic roundtrip fixtures, held out from tuning.
- Separate balanced per-encoding and realistic traffic-weighted reports; report exact normalization and ambiguity treatment.

## Fair product benchmark
- Compare frozen R21, one winning candidate, chardet pinned 7.0.1, latest released chardet separately, and charset-normalizer pinned/current.
- Beware chardet 7.0.1 is an older release; record accelerators and actual version/environment.
- Same bytes/run/runner; report Top-1, paired bootstrap interval by source, per-family/crawl failure rates, throughput, p50/p95 and independently measured RSS. No mixing results from different datasets.
- A winner must beat both competitors on unseen verified labels, with credible paired margin, avoid serious family regressions, and retain acceptable performance; otherwise do not claim launch-ready superiority.

## Bounded stop rule
Run one wider tournament, then one independent final product acceptance on a single precommitted champion. No post-hoc thresholds, crawl shopping, endless serial H experiments, or modifying frozen R21. User approval required for production release.
