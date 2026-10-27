# CodecCat vNext — Frozen ICU Source Confirmation Protocol (2026-10-27)

**This protocol is pre-registered before running the ICU benchmark.**

## Origin and license
- Independent test-text source: `unicode-org/icu`, `icu4c/source/test/testdata/csdetest.xml`.
- Immutable repository commit: `ad110a2381f1b75bfb875fb0b672b7c38dce4e4b`.
- Git blob: `076505fac2b2f313289207ddbdfaba9e9e99c3f6`.
- License: repository's Unicode License v3; individual test cases may have separate copyright notices. No text will be redistributed in the CodecCat repository; GitHub Actions retrieves the source for testing only.
- 33 XML test cases, with multiple explicitly marked encodings; it is an **ICU detector test suite**, not an unbiased sample of files from real usage.

## Non-negotiable isolation
- Train from `chardet/test-data` SHA `b0c0d206913ba118ffea2f7d306883e7a4a76db7` only. Never fit TF-IDF or encoding weights on ICU cases.
- Parse each `test-case` using standard Python XML parsing and join its text nodes, strip enclosing test indentation. Split the `encodings` attribute by space; the encoding name is before `/` (language suffix).
- Only accept codec labels recognized by Python and present in both candidate classifiers. Skip unencodable text without replacement. For each accepted encoding, create bytes by strict encoding the XML text; clip to 4096 bytes for every detector.
- Remove exact-byte or decoded-text overlaps with training samples. Retain all remaining case/codec combinations, but treat combinations sharing the same test-case text as correlated.
- No heuristic additions, model selection or parameter tuning based on this source. Do not re-use results as a subsequent untouched blind holdout.

## Locked comparisons
Compare: existing `stable23`, opt-in `experimental13`, `chardet==7.6.0`, `charset-normalizer==3.4.4`.
For each candidate report: exact codec label (canonical Python name), identical strict-decoded Unicode text, abstentions and wrong decoding. Compute both micro totals and group-by-original-case macro equality. Report source lengths and the number of excluded overlapping cases.
All detectors receive *identical* first 4096 bytes and strict finality consistent with clipping; if the original bytes exceed 4096, both expected and decoded text refer to the same nonfinal prefix.
Generated source examples have **no artificial BOM**. BOM-specific semantics are instead verified in dedicated unit tests.

## Pass/hold rule
No automatic model promotion on this one data source. An improvement must be source-disjoint, stable across source cases and codec families, and must not worsen robustness. If these conditions cannot be demonstrated, final validation stays blocked. Annual model and release artifacts stay unchanged.
