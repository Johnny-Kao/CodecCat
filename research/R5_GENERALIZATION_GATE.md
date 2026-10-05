# R5 Generalization Gate — 2026-10-06

## Status

**RELEASE BLOCKED ON ACCURACY GENERALIZATION**

CodecCat's clean runtime and package mechanics are ready enough for continued development, but the current fitted model should not be promoted as a release model yet.

The speed advantage generalized strongly across independent crawls. Accuracy parity with charset-normalizer did not.

## R5.0 — first independent single-model holdout

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37351567969

Development:
- canonical CodecCat development rows from CC-MAIN-2026-39
- single frozen CodecCat model

Independent holdout:
- CC-MAIN-2026-34
- n = 422
- fingerprint: `751e089484e0dcb325b987c1beefe10d0896df56bf10daa1018fc8d30e3cd60c`

Accuracy:

| Detector | Correct | Top-1 |
| --- | ---: | ---: |
| chardet 7.6.0 | 404 / 422 | **95.7346%** |
| charset-normalizer 3.5.2 | 382 / 422 | **90.5213%** |
| CodecCat v1 | 371 / 422 | **87.9147%** |

Median latency:

| Detector | µs/sample | Relative latency |
| --- | ---: | ---: |
| CodecCat | **127.10** | 1.00× |
| charset-normalizer | 912.67 | 7.18× |
| chardet 7 | 963.69 | 7.58× |

Interpretation:
- speed advantage generalized;
- CodecCat trailed charset-normalizer by ~2.61 pp;
- CC-MAIN-2026-34 was then explicitly reclassified as development data.

## R5.1 — data-only expansion + second untouched holdout

No runtime architecture, routing threshold, feature representation, specialist pair, or guard was changed.

Development:
- canonical CC-MAIN-2026-39 rows: 418
- additional CC-MAIN-2026-34 rows collected: 422
- combined unique development rows: 836
- development fingerprint:
  `8d27f0b5dd59180a584003e4a4fe77e8d2ca5e2bebd154334e24c7152821a35c`

Second untouched holdout:
- CC-MAIN-2026-30
- 3 exact byte overlaps with development removed before scoring
- evaluated n = 421
- fingerprint:
  `0d1b48f488ce90dc6d997d4bfad3f2e7c4d0289301baa561fb1cc57368351c44`

Authoritative run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37352132674

Frozen model:
- artifact: `codeccat-release-candidate-v2.npz`
- model SHA-256:
  `b50abf4448f7e634b1995cae7db6185992a176187006037ce0580c06b7b1632f`
- serialized size: 546,689 bytes
- artifact:
  https://github.com/Johnny-Kao/CodecCat/actions/runs/37352132674/artifacts/11362224777

Accuracy:

| Detector | Correct | Top-1 |
| --- | ---: | ---: |
| chardet 7.6.0 | 397 / 421 | **94.2993%** |
| charset-normalizer 3.5.2 | 392 / 421 | **93.1116%** |
| CodecCat v2 | 371 / 421 | **88.1235%** |

CodecCat gaps:
- vs charset-normalizer: **-4.99 pp**
- vs chardet 7: **-6.18 pp**

Median latency:

| Detector | µs/sample | Relative latency |
| --- | ---: | ---: |
| CodecCat | **80.04** | 1.00× |
| charset-normalizer | 795.38 | 9.94× |
| chardet 7 | 852.94 | 10.66× |

## Decision

### What is now established

1. **Runtime architecture is not the problem.**
   - clean package exactly reproduces locked P15;
   - speed advantage persists and is large on two separate independent crawls.

2. **Simple data-volume expansion is not sufficient.**
   - adding CC-MAIN-2026-34 to development did not restore parity on CC-MAIN-2026-30.

3. **The active research bottleneck is generalization of ranking/calibration across crawls.**
   - route/scorer representation may still be useful;
   - the failure must be decomposed before changing architecture.

4. **Do not resume P17-style micro-optimization.**

5. **Do not release the current model as if charset-normalizer parity were established.**

## Next research layer

The next research session should be **R6 — cross-crawl error decomposition**, not another blind model fit.

R6 questions:

1. Which encoding families account for the CodecCat-vs-charset-normalizer gap?
2. Are the misses:
   - base scorer ranking failures;
   - calibration/reranker failures;
   - route assignment failures;
   - unsupported/ambiguous label normalization;
   - insufficient family-specific evidence?
3. Does the ground truth usually remain in CodecCat Top-3/Top-5?
4. Are errors stable across CC-MAIN-2026-34 and CC-MAIN-2026-30?
5. Can one small, cross-crawl-safe mechanism close a material share of the gap without heuristic growth?

Methodological boundary:
- CC-MAIN-2026-30 remains release-gate evidence and must **not** be used for tuning unless it is explicitly reclassified as development data.
- If R6 requires tuning against CC-MAIN-2026-30, a new untouched crawl must be reserved first.

Recommended first R6 move:
- analyze already-development CC-MAIN-2026-34 in detail;
- keep CC-MAIN-2026-30 untouched;
- only propose changes that can be selected from CC-MAIN-2026-34 + prior development evidence;
- validate once on a newly reserved crawl after the change is frozen.
