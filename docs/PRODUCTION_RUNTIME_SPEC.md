# Production Runtime Specification

Updated: 2026-10-06

## Purpose

This document translates the accepted CodecCat research path into a clean runtime contract.

CodecCat is a standalone character-encoding detector. Its external comparators are:

1. charset-normalizer
2. chardet 7

Neither comparator is an upstream.

## Important model boundary

The canonical 365/418 result is a pooled 4-fold held-out result. Each fold uses a different fitted model.

Therefore:

- the fold models are validation fixtures, not release models;
- no single fold model may be shipped as the CodecCat production model;
- runtime reconstruction is validated against those fold fixtures;
- a release model must later be trained on the full eligible training set;
- the release model must be evaluated on a new independent holdout that was not used for model selection or calibration.

## Runtime pipeline

```text
bytes
  -> route (U / N / RL / RH)
  -> H/M/T-768 sampling
  -> 518-dim feature vector
  -> route-local fused linear scorer
  -> stable descending rank
  -> deterministic rule rerank
  -> candidate calibrator
  -> triad gate
  -> UTF-8 / UTF-8-SIG specialist
  -> UTF-8 / GB18030 specialist
  -> replacement-rate guard
  -> ranked encodings
```

## Routing

Only the first 4096 bytes are used for route selection.

1. strict UTF-8 decodes -> `U`
2. otherwise contains NUL -> `N`
3. otherwise high-byte ratio <= 0.02 -> `RL`
4. otherwise -> `RH`

The 0.02 residual split survived external WARC validation and replaced the earlier provisional 0.03 cut.

## H/M/T sampling

Inputs <= 768 bytes are used directly.

Longer inputs use:

- first 256 bytes;
- middle 256 bytes;
- last 256 bytes.

This keeps the scoring representation fixed-cost with respect to large inputs.

## Feature vector

Dimension: 518

- 0:256 — normalized unigram histogram
- 256:512 — normalized wrapped uint8 adjacent-byte sum histogram
- 512 — log-scaled sample length
- 513 — high-byte ratio
- 514 — NUL ratio
- 515 — printable ASCII ratio
- 516 — LF ratio
- 517 — CR ratio

Accepted P15 implementation details:

- preallocate float32 output;
- assign integer counts into output then multiply in-place by reciprocal;
- adjacent uint8 addition intentionally wraps modulo 256;
- use `np.add.reduceat` for printable/high aggregate ranges;
- fused route scorer uses `np.dot(w, x) + b`.

## Downstream policy

The downstream stack is deliberately narrow.

- deterministic rerank handles BOM, strict UTF-8, ASCII, and the validated ISO-8859-1 correction;
- candidate calibrator considers only the leading candidate set;
- triad specialist is restricted to UTF-8 / CP1251 / GB18030;
- pair specialists are restricted to UTF-8 vs UTF-8-SIG and UTF-8 vs GB18030;
- the GB18030 -> UTF-8 correction is blocked when UTF-8 replacement rate exceeds 0.02.

No additional heuristic should be added without independent held-out evidence.

## Correctness contract

For reconstruction against the canonical fold fixtures:

- evaluated = 418
- hits = 365
- Top-1 = 87.3206%
- final ranking mismatch versus locked P15 = 0

Any clean-runtime mismatch is a bug until shown otherwise.

## Performance contract

The locked research runtime is P15.

P16 found no exact-equivalent final byte-path candidate that improved full-pipeline runtime, so routine micro-optimization is closed.

Clean package implementation must first preserve correctness. If it materially regresses relative to P15, profile the clean implementation and recover the lost mechanism without reintroducing experiment-layer complexity.

## Release-model contract

A release model is a separate artifact from the CV fixtures.

Required sequence:

1. freeze clean runtime;
2. build a new independent evaluation corpus;
3. fit final parameters on all eligible training data;
4. evaluate the final artifact once on the independent corpus;
5. compare the same artifact against charset-normalizer and chardet 7;
6. only then ship the model with the package.

This separation prevents cross-validation evidence from being mistaken for deployment evidence.
