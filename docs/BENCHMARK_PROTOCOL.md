# Benchmark Protocol

## Comparators

CodecCat is compared against exactly two external packages:

1. `charset-normalizer`
2. `chardet 7`

They are references, not upstreams.

## Evidence layers

### A. Cross-validation development benchmark

Purpose:
- validate architecture and calibration changes;
- compare clean runtime against the locked research runtime;
- compare the three detectors on identical held-out fold rows.

This layer may use the canonical 418-row evaluation corpus.

Claims must explicitly say that CodecCat uses fold-specific held-out models.

### B. Release-model benchmark

Purpose:
- measure the single artifact users will actually install.

Requirements:
- one frozen CodecCat release model;
- independent holdout not used during architecture, feature, threshold, calibration, or model selection;
- same bytes and normalized labels for all three systems.

Only this layer may support release-facing claims such as "CodecCat is X% accurate" or "CodecCat is Yx faster than ...".

## Accuracy

Primary metric:
- Top-1 accuracy.

Secondary diagnostic metrics:
- Top-3 / Top-5 when the compared API exposes meaningful ranking;
- per-family accuracy;
- input-size buckets;
- route buckets;
- confusion pairs.

Do not combine accuracy and runtime into one score for public claims.

## Runtime

Measure model construction separately and exclude it from steady-state detection latency.

For each system:
- warm up once;
- run multiple full-corpus passes in the same process;
- report median ns/sample;
- retain min/max or dispersion;
- keep thread-count environment fixed;
- use identical input ordering.

Required workload views:
- overall;
- short <=256 bytes;
- medium 257-4096 bytes;
- large >4096 bytes;
- UTF-8-heavy;
- legacy/non-UTF8.

GitHub Actions public runners remain the shared benchmark environment for project evidence.

## Versioning

Every report must record:
- CodecCat commit/model fingerprint;
- charset-normalizer version;
- chardet version;
- Python version;
- NumPy version;
- corpus fingerprint;
- runner OS.

A benchmark becomes historical evidence when any of those materially changes.
