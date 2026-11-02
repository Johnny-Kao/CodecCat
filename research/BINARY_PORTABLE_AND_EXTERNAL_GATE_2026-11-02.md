# CodecCat binary-model release gates — 2026-11-02

## Implemented in research branch
- Safe portable model schema `vnext-l2-tfidf-binary12-sgd-huber-v3`; preserves legacy Stable23 loader, rejects unsupported model profiles. No changes to main, alpha release or PyPI.
- [GitHub Actions roundtrip #41368432314](https://github.com/Johnny-Kao/CodecCat/actions/runs/41368432314): PASS; same predictions across in-memory, restored and new-process loaders, 1138/1138 on fold zero for both binary and legacy. Binary archive 4,850,308 bytes; legacy archive 10,973,355 bytes.
- [GitHub Actions external #41368566768](https://github.com/Johnny-Kao/CodecCat/actions/runs/41368566768): PASS; BYVoid/uchardet fixtures with exact raw-byte and decoded-text overlaps excluded: 20 cases, Stable23 19/20, binary12 19/20, binary-only 1, Stable23-only 1; 36 files excluded for overlap. Labels originate in filenames and this is not a large blind source-independent benchmark.
- Earlier [paired candidate #41362464702](https://github.com/Johnny-Kao/CodecCat/actions/runs/41362464702): binary12 2882/3086 vs Stable23 2828/3086, ~1.75 pp gain on chardet-owned development cases; CPU p50 lower, encoding-family regressions remain.

## Decision
Binary12 is now serializable and has no *net* accuracy regression on small uchardet external sample, but **does not demonstrate independent superiority**. Fail release-promotion gate until larger, authentic, source-diverse holdout with stable codec truth and confidence intervals exists; 20 observations do not support generalization. Candidate can remain experimental, main Stable23 untouched.

## Scope and caveats
The portable model loader is modified but has not been tested as installed release wheel/API with all platforms. Archive-size reduction doesn't imply same relative RAM reduction. The validation used prefix decoding and selected original code-page labels, not all real-world datasets. Future clean-wheel end-to-end test and source-provenance audit remain.
