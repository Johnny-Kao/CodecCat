# CodecCat short-input recall: research checkpoint (2026-11-05)

Status: **No accuracy improvement accepted. Research branch only.**

## Locked evaluation contract

- Five disjoint panels, fixed seed `20261105`; 3,092 distinct raw-byte fixture records.
- Source split: chardet 3,003, Enca 81, charset-normalizer 8.
- Group-held-out training on chardet subsets; external fixtures use full chardet-trained model.
- Identical full-document strict decoding / leading U+FEFF normalization.
- Baseline: 2,914/3,092 (94.243%); chardet 7.5.0 comparison 2,977/3,092 (96.281%).
- These previously used development fixtures **are not an untouched independent test**.
- Long inputs >256 bytes must remain byte-for-byte prediction invariant under any short-only route.

## Error analysis

[CI 41964906140](https://github.com/Johnny-Kao/CodecCat/actions/runs/41964906140)

Of 188 short (<=256 bytes) input records:
- 140 correct; 48 wrong.
- 31 wrong with no equal-text interpretation in the current top-three selected candidates.
- 16 wrong due to ranking / L3 not selecting the accessible candidate.
- 1 wrong because L3 overrides correct L2.
- Every one of the 188 input records has an equal-text decoding in at least one trained class.
- 172/188 have an equal-text interpretation within the classifier's first 16 ranks (a theoretical candidate-recall bound, **not** expected top-1 accuracy).
- 175/188 original truth codec labels belong to trained classes; label equality is not required when different codecs yield identical Unicode text.
- Enca: 28 of the 31 top-three-unreachable failures. This is a strong distribution/source issue, not evidence that a particular codec alias needs hardcoding.

## Tested options

| Option | Overall | Short <=256 | Enca | Verdict |
|---|---:|---:|---:|---|
| Frozen baseline | 2914/3092 | 140/188 | 45/81 | Retain |
| 256-byte-only L2 trained on truncated rows | 2899/3092 | 125/188 | 43/81 | Reject |
| 256-byte L2 with full-text + 128/256 prefix augmentation | 2910/3092 | 136/188 | 46/81 | Reject |
| Trigram quality-gated top-three reranker | 2908/3092 | — | 45/81 | Reject |
| Binary 1-3 gram L2 | 2867/3092 | 135/188 | 46/81 | Reject |
| Nonbinary 1-2 gram L2 | 2893/3092 | 135/188 | 43/81 | Reject |
| Short-only C1/Unicode anomaly reranker | 2914/3092 | 140/188 | 45/81 | Reject (no gain) |

Short-only C1/Unicode guard:
[CI 41965193377](https://github.com/Johnny-Kao/CodecCat/actions/runs/41965193377).
It changed eight predictions, none of which changed whether decoded text matched truth.

## Next architectural question

The remaining obstacle is not **length detection** (an entry-point length check is trivial),
and not **missing codec classes** (all 188 have some trained equal-text decoding).
The obstacle is source-generalizable **selection among many structurally plausible decodings**.

Do not increase candidates or add codec-specific exceptions merely to optimize these fixtures.
Before another substantial architecture change:
1. Collect a genuinely new, provenance-audited multilingual corpus with original bytes and verified encodings, split by source/document family.
2. Reserve an untouched confirmation cohort.
3. Compare candidate scoring / word plausibility on development data with paired fixes/breaks and source/length buckets.
4. Only then run one frozen validation on the confirmation cohort; require >=95% consistently across prespecified mixed panels with explicit small-source uncertainty.
5. Treat performance and memory optimization as a separate gate after accuracy stability.

No main, Alpha, published model, or production inference code changed.
