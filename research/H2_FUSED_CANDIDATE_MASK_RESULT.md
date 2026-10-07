# H2 Fused Candidate-Mask A/B Result

Date: 2026-10-07
Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/37620757672
Status: **REJECT raw-score masking integration**

## Result

H1 structural validity itself remained safe:

- pooled development rows: 521
- covered ground-truth encodings masked impossible: **0**

However, applying the structural mask by replacing raw route-local scores with a very negative value **before** the existing downstream policy was harmful:

- baseline: 505 / 521 = 96.93%
- masked: 474 / 521 = 90.98%
- delta: **-5.95 pp**
- changed Top-1: 31
- beneficial: 0
- harmful: 31

By crawl:

- CC-MAIN-2026-34: 253 -> 240, -13 hits
- CC-MAIN-2026-39: 252 -> 234, -18 hits

The damage concentrated in RH.

## Interpretation

This does **not** invalidate H1 structural rules.

The cause is architectural:

1. the calibrator and specialists were trained on the original raw-score geometry;
2. replacing selected raw scores with an extreme sentinel changes margins and downstream feature values;
3. downstream models therefore receive out-of-distribution score inputs;
4. they can change decisions even though the true encoding itself was never structurally masked.

Therefore structural impossibility must not be injected by mutating the learned score vector unless downstream models are explicitly redesigned/retrained for masked score semantics.

That redesign would add coupling and retraining burden, which conflicts with the current vNext maintenance objective.

## Runtime finding

The fused pure-Python five-state scan was also not production-viable:

- median mask-only cost: ~517 us/sample

This is materially larger than the current CodecCat release runtime on R22 (~164 us/sample).

The prototype is research evidence only.

## Decision

Reject:

```text
raw learned scores
-> set impossible candidates to -inf
-> existing downstream stack
```

Do not proceed to H3 from this integration.

Retain H1 as a validated structural-invariant library concept.

## Next research direction

The next bounded experiment should test **late structural veto semantics**:

```text
existing frozen learned pipeline
-> proposed final candidate
-> if candidate is structurally impossible:
       choose the highest-ranked structurally possible alternative
-> otherwise:
       preserve output exactly
```

This has two advantages:

- no perturbation of learned score geometry;
- zero behavioral change unless the final learned choice violates a specification-derived invariant.

Promotion requirements remain strict:

- zero false-impossible on covered ground truth;
- zero harmful changes across independent development crawls;
- implementation cost must be reduced substantially before production consideration.

Because H1 observed no structurally impossible final Top-1 predictions on the current development set, late veto may intentionally be a no-op on current data. That is acceptable: its purpose is correctness protection, not benchmark fitting.
