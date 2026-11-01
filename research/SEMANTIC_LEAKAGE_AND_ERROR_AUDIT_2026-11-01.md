# CodecCat semantic leakage and pinned comparator study — 2026-11-01

**Status:** Completed GitHub Actions run; development-only, no release gate.
**Run:** https://github.com/Johnny-Kao/CodecCat/actions/runs/41279564156
**Data:** `BYVoid/uchardet@4e685757780cb3c652fc6c9ec759f62888969ec9`, training `chardet/test-data@b0c0d206913ba118ffea2f7d306883e7a4a76db7`.

## Cross-corpus leakage

57 strictly-decodable fixture files were evaluated. **36/57** have a normalized decoded-text SHA-256 that occurs in the development training corpus, despite only **33/57** being byte-identical in the earlier audit. Thus at least three more cross-encoded or otherwise byte-different text duplicates were discovered. Only 21 remain for exploratory comparison. This is *not* exhaustive deduplication: normalization only covers the first 4096 bytes, line endings and leading BOM; near duplicates and shared passages are not excluded.

## Paired comparison, 21 nonidentical-text samples

| System | Strict decoded text correct | Notes |
|---|---:|---|
| stable23 + residual, retrained at run-time | **20/21** | Not the serialized published Alpha |
| chardet **7.0.1** | **19/21** | Fixed dependency |
| charset-normalizer **3.4.4** | **21/21** | Fixed dependency |

The sole CodecCat error is `test/ja/shift_jis.txt`: true file label `shift_jis`, predicted `cp1006`, top-three `cp1006/cp864/cp865`. chardet selected `cp720`; charset-normalizer selected `cp932` and decoded the reference text successfully. The other differentiated error is chardet's Big5 fixture, predicted `cp1140`, while CodecCat and charset-normalizer decoded the reference text correctly.

## Interpretations and limitations

- The Japanese Shift-JIS case is a **candidate routing/feature-model weakness**, not proof of a solution. Do not patch a heuristic based on one fixture.
- UTF-16 and similar prefix decoding can introduce artifacts; this study compares same strict decoded *prefix* with BOM removal and does not certify entire documents or original encoding provenance.
- Repository fixtures represent a single upstream suite and may re-encode identical text, so 21 rows are not 21 independent original-language documents.
- Filename-derived labels are not individually verified provenance.
- More independent original-source documents, especially CP037/CP500 and Japanese, are required to determine whether any feature/model change generalizes.
- No model promotion, API change, or claim of statistical superiority.

## Next engineering decision

Keep unchanged Alpha. A source-diverse batch with leakage filtering and paired competitor version pinning is now the acceptance requirement before touching stable23 features or residual weights.
