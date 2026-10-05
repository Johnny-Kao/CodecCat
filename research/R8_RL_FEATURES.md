# R8 RL Compact Feature Tournament — 2026-10-06

## Trigger

R7 rejected generic RL sampling expansion. R8 tested whether the first 4096 bytes already inspected by routing contain compact high-byte statistics that could improve RL discrimination without changing HMT768.

Authoritative paired run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37359566455

## Candidates

All arms used identical reconstructed rows, fold assignment, legacy rows, and fully refitted downstream models.

- baseline: locked 518-dim HMT768
- RL high16: +16 normalized bins over 0x80..0xFF
- RL highstats: +4 full-4K summary signals
  - high-byte ratio
  - C1-range share
  - normalized high-byte entropy
  - concentration
- RL high16 + stats: +20 features

Extra dimensions were zero for non-RL inputs.

## Paired results

Baseline:
- 365/417 = **87.52998%**
- RL: 35/40 = **87.50%**

RL high16:
- 362/417 = 86.8106%
- overall delta: **-3 hits / -0.7194 pp**
- RL delta: **0 hits / 0.00 pp**

RL highstats:
- 363/417 = 87.0504%
- overall delta: **-2 hits / -0.4796 pp**
- RL: 34/40 = 85.00%
- RL delta: **-1 hit / -2.50 pp**

RL high16 + stats:
- 362/417 = 86.8106%
- overall delta: **-3 hits / -0.7194 pp**
- RL delta: **0 hits / 0.00 pp**

## Decision

**All R8 feature augmentations are rejected.**

Conclusions:
1. RL weakness is not fixed by simply exposing coarse first-4K high-byte distribution statistics.
2. The locked HMT768 representation remains production baseline.
3. Do not spend untouched holdout evidence on any R8 candidate.
4. Small feature-width changes can perturb downstream score distributions even when non-RL appended dimensions are zero; paired full-pipeline evidence remains mandatory.
5. Next research step is not another arbitrary feature family.

## R9 direction

R9 must first decompose RL itself:
- truth/prediction confusion pairs;
- raw truth Top-1/Top-3/Top-5 position;
- per-class RL training support;
- whether final errors are caused by rank retrieval or wrong ordering inside an already-correct candidate set.

Only after that decomposition should one of these be tested:
- RL-only specialist over the existing top candidates; or
- RL-specific class-prior / sample-weight correction.

No production change is authorized yet.

`CC-MAIN-2026-30` remains untouched.
