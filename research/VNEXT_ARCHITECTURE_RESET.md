# vNext architectural reset after Gate 3

**Decision:** Gates 1 and 3 largely measured R21 variants; Gate 2 used new weights but retained R21's original feature information and a small weakly labelled corpus. None is proof of a full independent Logic-First architecture. R21 stays frozen as a comparator, not the pipeline to be improved.

## Independent prototype (not release-ready)
`src/codeccat/vnext_independent.py` exposes independent `fit(samples)` and `TransitionModel.rank(bytes)`. It has NO import of R21's Runtime, model weights, route selector, specialists, or HMT768. In particular its directional adjacent-byte transition representation addresses the earlier information-loss concern; its log-likelihood is trained from user-supplied labelled byte corpora. The standard byte-sequence validator supplies negative evidence only.

The prototype intentionally demonstrates architecture separation, not competitive accuracy. There is no trained production model or independently verified holdout. It currently has 2,048 hashed transition dimensions and Naive-Bayes-like scoring. Known limitations: hashing collisions, length sensitivity, class imbalance, unassigned-codepoint viability, ambiguous ASCII, unseen labels, and no stable model bundle format. These should be measured, not described as solved.

## Remaining test budget
1. Gate 4: ONE large paired benchmark of **independent** architectures (not R21 modifications), including separate training and held-out source families, multiple language-likelihood estimators + candidate-safety policies, metadata/no-metadata views, accuracy by encoding and source, p50/p95 latency, RSS, model bytes and fit cost. R21 and newest chardet/charset-normalizer are controls only. Use no untouched final acceptance source for tuning.
2. Gate 5: ONE frozen-candidate evaluation on truly independent verified-label sources if available. If unavailable, report BLOCKED, stop research tests and do not claim competitive success.

Do not spend Gate 4 just proving a trivial prototype runs; focus on its multiple source-disjoint alternatives and data provenance.
