# Production Runtime Specification

Updated: 2026-10-07

## Purpose

This document defines the frozen CodecCat runtime architecture after the R19 parameter-suitability audits and R20 cached-path validation.

CodecCat is a standalone character-encoding detector. Its external comparators are:

1. `charset-normalizer`
2. `chardet 7`

Neither comparator is an upstream.

## Model boundary

Cross-validation fixtures are validation artifacts, not release models.

The current development evidence is pooled across held-out folds. No fold-specific model may be shipped.

A release model must be trained once on the full eligible development corpus after architecture freeze, serialized as one immutable artifact, and evaluated exactly once on a fresh independent crawl.

## Frozen runtime architecture

```text
bytes
  -> route once (U / N / RL / RH)
  -> H/M/T-768 once
  -> 518-d baseline features once
  -> baseline route-local scorer
  -> downstream baseline policy
  -> raw top1-top2 margin
  -> route-local ambiguity gate
       -> if U/RH and ambiguous:
            compute 368-d generic order extras from cached H/M/T
            -> 886-d full-B route-local scorer
            -> full-B downstream policy
       -> otherwise:
            keep baseline result
  -> RL-only specialist on the baseline path
  -> ranked encodings
```

The runtime MUST NOT recompute route, H/M/T sampling, or baseline features after the first pass.

## Frozen routing

Only the first 4096 bytes are used for route selection.

1. strict UTF-8 decodes -> `U`
2. otherwise contains NUL -> `N`
3. otherwise high-byte ratio <= 0.02 -> `RL`
4. otherwise -> `RH`

The S3 threshold is frozen at `0.02`.

## Frozen sampling

Inputs <= 768 bytes are used directly.

Longer inputs use:

- first 256 bytes;
- middle 256 bytes;
- last 256 bytes.

This H/M/T-768 sample is computed once and reused by both baseline and optional full-B feature extraction.

## Baseline representation

Dimension: 518.

- 0:256 — normalized unigram histogram
- 256:512 — normalized wrapped uint8 adjacent-byte sum histogram
- 512 — log-scaled sample length
- 513 — high-byte ratio
- 514 — NUL ratio
- 515 — printable ASCII ratio
- 516 — LF ratio
- 517 — CR ratio

## Full-B representation

Dimension: 886.

The first 518 dimensions are exactly the baseline representation.

The 368 conditional dimensions are:

- 256 bins — adjacent-byte modular difference histogram;
- 64 bins — upper-6-bit XOR histogram;
- 48 bins — position-aware high-byte distribution, 16 bins for each H/M/T segment.

Full-B extras are eligible only on routes `U` and `RH`.

R16A established bit-exact equivalence between the fused implementation and the accepted full-B reference.

## Frozen downstream policy

The downstream stack remains deliberately narrow.

- deterministic rerank handles BOM, strict UTF-8, ASCII, and the validated ISO-8859-1 correction;
- candidate calibrator considers only the leading candidate set;
- triad specialist is restricted to UTF-8 / CP1251 / GB18030;
- pair specialists are restricted to UTF-8 vs UTF-8-SIG and UTF-8 vs GB18030;
- the GB18030 -> UTF-8 correction is blocked when UTF-8 replacement rate exceeds 0.02;
- the R12 specialist is RL-only.

No new heuristic or specialist may be added before a fresh independent evaluation unless the architecture freeze is explicitly reopened.

## Frozen suitability parameters

These are architecture-level choices, not values selected from the future release holdout.

- route scorer logistic `C = 0.5`
- R12 specialist threshold = `0.65`
- ambiguity gate training quantile = `0.50`
- full-B eligible routes = `U`, `RH`
- H/M/T sample width = 768
- S3 high-byte threshold = `0.02`

The ambiguity gate stores fitted route-specific raw-margin thresholds in the release artifact. Those thresholds are learned only from the full eligible development corpus.

## Development correctness contract

R19C and R20B define the frozen development behavior.

Pooled development fixture:

- evaluated = 418
- hits = 378
- Top-1 = 90.4306%
- escalated = 249 / 418
- prediction mismatch between cached production path and frozen reference = 0
- escalation mismatch = 0

Route totals at the frozen center:

- RH: 229 / 257
- RL: 37 / 41
- U: 112 / 120

`CC-MAIN-2026-30` was not used for R19/R20 tuning or validation and remains excluded from parameter selection.

## Parameter suitability evidence

R19C tested the final cascade at route-scorer `C` values 0.4 / 0.5 / 0.6 with all non-C policy fixed.

Result:

- 0.4 -> 378 / 418
- 0.5 -> 378 / 418
- 0.6 -> 378 / 418

Accuracy range: 0 hits.

This is evidence that `C = 0.5` is a stable operating point rather than a development-set knife edge.

R12 threshold 0.65 also passed its local suitability audit.

## Performance contract

R20A established bit-exact cached feature execution:

- route mismatch = 0
- baseline feature mismatch = 0
- full-B feature mismatch = 0

Same-run feature-path medians:

- route + cached baseline: ~33.95 us/sample
- route + cached conditional full-B: ~46.41 us/sample
- reference full-B everywhere: ~52.18 us/sample

R20B established end-to-end exact behavior with the cached production path.

Same-run median-across-fold latency:

- frozen reference path: ~1569.7 us/sample
- cached production path: ~1189.9 us/sample

Observed reductions:

- median: ~24.2%
- p95: ~23.6%
- p99: ~25.1%

These numbers are development-run evidence only. Public release performance claims require the release-model benchmark on the independent holdout.

## Release-model contract

The release artifact must contain:

1. baseline route-local scorer models;
2. full-B route-local scorer models for U/RH;
3. baseline downstream calibration/specialist models;
4. full-B downstream calibration/specialist models required by the frozen path;
5. RL specialist model;
6. fitted U/RH ambiguity-margin thresholds derived from the full development corpus;
7. class/family metadata;
8. model schema/version metadata.

The release artifact must not contain fold-specific state or holdout-derived thresholds.

## Release sequence

Required order:

1. architecture freeze;
2. clean runtime implementation of the frozen cascade;
3. exact equivalence against R20 development fixtures;
4. fit one release model on all eligible development data;
5. serialize and fingerprint the artifact;
6. acquire a fresh Common Crawl holdout not previously used for model/parameter selection;
7. remove any exact byte overlap with development data;
8. evaluate the frozen artifact once;
9. compare the same bytes against charset-normalizer and chardet 7;
10. do not tune on that holdout.

If the fresh holdout causes any architecture, threshold, feature, calibration, or model-selection change, that holdout becomes development data and cannot support the release claim.

## Release prohibition

Architecture freeze does not authorize:

- merge to `main`;
- PyPI publication;
- package release;
- public release claims.

Those actions require explicit approval after the independent holdout gate.
