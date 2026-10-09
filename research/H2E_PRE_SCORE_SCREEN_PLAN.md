# H2e Pre-Score Structural Screening — Preregistered A/B

Date: 2026-10-09
Status: preregistered research
Release line affected: none

## Question

Does changing only the execution order from:

```text
all route candidates -> frozen learned scorer
```

to:

```text
spec-derived structural screen -> surviving candidates -> same frozen learned scorer
```

improve or preserve raw scorer accuracy while reducing the number of learned score rows that need evaluation?

## Fixed inputs

- Frozen R21 model artifact and parameters.
- Development crawls only: CC-MAIN-2026-39 and CC-MAIN-2026-34.
- R22 / CC-MAIN-2026-25 is excluded.
- Existing 518-d baseline representation.
- Existing route assignment.
- No retraining.
- No C / gate / R12 / feature / threshold changes.

## Structural screen

Use only the H1 permissive specification-derived byte-grammar validators:

- Big5
- GB18030
- Shift-JIS
- EUC-JP
- EUC-KR

A candidate is removed only when the observed byte prefix contains an explicit structural violation for that encoding.

The stricter Python codec mapping-table behavior from H2d is **not** used in this experiment.

## A/B

A — baseline raw route scorer:

```text
518-d feature vector
-> evaluate every route-model class row
-> raw top1
```

B — pre-score screen:

```text
structural validity mask
-> evaluate/rank only surviving route-model class rows
-> raw top1
```

The same frozen weight and bias rows are used in both arms.

## Metrics

Primary:

1. raw top1 accuracy by crawl and pooled;
2. beneficial / harmful top1 changes;
3. ground-truth false elimination;
4. candidate-row reduction:
   - rows before;
   - rows after;
   - fraction removed;
5. route-level consistency.

Secondary:

- distribution of removed structural candidates by encoding;
- score-kernel row-operation reduction as a compute proxy.

## Interpretation

Positive architectural evidence requires:

- zero observed ground-truth false elimination;
- no accuracy loss in either development crawl;
- non-zero candidate-row reduction;
- no parameter adjustment.

A gain is stronger evidence but is not required. A no-op with meaningful row reduction can still support the ordering thesis.

A failure does not authorize threshold tuning or sample-specific rules.

## Non-goals

- no production integration;
- no downstream calibrator mutation;
- no specialist fitting;
- no R22 analysis;
- no strict-codec semantic expansion;
- no benchmark-specific rule selection.
