# CodecCat oracle ceiling and conditional bigram gate — 2026-11-05

## Decision

**Retain current 2,914/3,092 (94.243%) layered research baseline.**
**Reject the encoding-conditional byte-bigram L3 candidate (2,882/3,092).**
Neither experiment modifies main, alpha, or the published detector.

## Oracle ceiling audit

[GitHub Actions 42002912017](https://github.com/Johnny-Kao/CodecCat/actions/runs/42002912017), PASS.

The 178 current errors on the five fixed disjoint panels decompose as follows:

| Error class | Count |
|---|---:|
| Correct decoded text is already in strictly-valid L2 Top 3 but ranked wrong | 117 |
| Best correct equivalent is in valid ranks 4–16 | 37 |
| Correct equivalent is in deeper ranked candidates | 17 |
| L2 top choice is correct but a later layer overrides it incorrectly | 6 |
| No trained-class codec yields equal decoded Unicode | 1 |
| **Total errors** | **178** |

There is **no contradiction** between 177/178 errors being theoretically fixable with a perfect oracle and 95% remaining difficult: the oracle sees ground truth, whereas inference does not.

Top-k valid-score error coverage: Top-2 88; Top-3 123 (includes 6 post-L2 mistakes); Top-5 144; Top-8 151; Top-16 160; Top-32 164; all trained classes 177. Ranking-only and L1/L3 interventions cannot be inferred solely from these upper bounds.

Source distribution of errors: chardet 141, Enca 36, charset-normalizer 1. Of 178 errors, 48 are in <=256 bytes, 130 in >256 bytes.

To cross **95% overall**, current 2914 correct requires **24 net additional** correct items on these 3092 test cases. For **all five** panels individually >=95%, required *per-panel* net improvements from baseline are 12, 2, 8, 1, 3 = **26** (distributed appropriately). Neither proves random-sample steadiness or independent generalization.

## Conservative independent reranker A/B

[GitHub Actions 42003429036](https://github.com/Johnny-Kao/CodecCat/actions/runs/42003429036), PASS.

A per-encoding byte-bigram background-normalized log-likelihood ratio model was trained exclusively on the permitted training groups and consulted only for the L2 Top-3 candidates under a prespecified low-margin gate. The existing language L3 override was preserved.

| Metric | Baseline | Candidate |
|---|---:|---:|
| Overall | 2914/3092 (94.243%) | 2882/3092 (93.208%) |
| Short <=256 | 140/188 | 137/188 |
| Long >256 | 2774/2904 | 2745/2904 |
| Enca | 45/81 | 45/81 |
| Paired fixes / breaks | — | 16 / 48 |

All five panels worsened. **Reject** as uncalibrated extra evidence; merely having a distinct codec-conditional scoring signal does not establish that it corrects the L2 score.

## Next useful experiment

A strictly out-of-fold **pairwise candidate discriminator** trained on source-group-separated training only (features: L2 margin, strict-validity, decoded-script features, language coherence, and byte-bigram evidence), with:
1. No evaluation data used in fitting, threshold selection, or error-targeted rules.
2. Candidate ordering learned on training-only inner folds; avoid training a meta-ranker on its own base-model in-sample scores.
3. Separate gains/losses by source and <=256-byte bucket.
4. A fresh, provenance-audited independent confirmation corpus before any claim of robust >=95%.

No architectural promotion is authorized from the above development results.
