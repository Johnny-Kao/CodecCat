# CodecCat vNext Hybrid Research Contract

Date: 2026-10-07
Status: research only
Release line affected: none

## Decision

The current R20/R21 release candidate remains frozen.

Fresh independent evidence on CC-MAIN-2026-25 showed:

- CodecCat: 233/258 = 90.31%
- charset-normalizer 3.5.2: 234/258 = 90.70%
- chardet 7.6.0: 242/258 = 93.80%
- CodecCat median latency: ~164 us/sample
- charset-normalizer: ~932 us/sample
- chardet 7: ~1175 us/sample

The release candidate must not be changed in response to those holdout errors.

CC-MAIN-2026-25 is now release-evaluation evidence. It must not be used to choose vNext thresholds, rules, features, confusion groups, or models.

## vNext thesis

CodecCat should not chase additional accuracy by repeatedly fitting more parameters to the same benchmark distribution.

The next research stage should test a hybrid design:

```text
offline / infrequent work
    -> train compact parameter tables from broad data
    -> freeze parameters

online / hot path
    -> cheap structural rules and validity checks
    -> eliminate impossible or implausible candidates
    -> use existing learned scores
    -> invoke a small specialist only when ambiguity remains
```

The target is not a purely learned detector and not a large heuristic rule engine.

The intended architecture is:

```text
stable protocol knowledge
+ generic statistical scoring
+ sparse ambiguity specialists
```

## Maintenance objective

CodecCat should be able to remain useful for many years without annual benchmark-specific rule growth.

A production mechanism should therefore have one of these justifications:

1. **encoding-spec invariant**
   - byte legality
   - code-unit structure
   - BOM or escape semantics
   - impossible continuation/range combinations

2. **generic statistical invariant**
   - a representation or score property expected to hold across corpora

3. **narrow confusion mechanism with a causal explanation**
   - two encodings are structurally confusable for a known reason
   - a specialist operates only after a stable ambiguity condition

A mechanism should not be accepted solely because it corrects several samples in one corpus.

## Training philosophy

Training is allowed and remains central, but parameter fitting is treated as an infrequent offline operation.

The intended trade-off is:

- expensive corpus work can happen offline;
- production inference should use the resulting compact parameters cheaply;
- deterministic structural logic can handle facts that do not need to be relearned;
- the model should not be retrained merely because a yearly benchmark distribution changes.

The default expectation is that a stable release may live for years without parameter updates.

## Research ordering

### Stage H1 — structural candidate elimination

Research only specification-derived checks that can remove impossible encoding candidates before expensive ambiguity resolution.

Examples of acceptable questions:

- Is this byte sequence structurally impossible under Big5?
- Is it impossible under GB18030?
- Are UTF-16/32 alignment and null patterns inconsistent with a candidate?
- Does an encoding require byte ranges or pair structure that the sample violates?

Requirements:

- no threshold tuning from R22;
- no encoding-specific rule unless supported by the encoding specification;
- cost must be bounded and measured;
- preferably reuse bytes already scanned by routing/HMT.

### Stage H2 — candidate-mask integration

Test whether a structural-validity mask can be combined with existing model scores without retraining the full architecture.

Preferred form:

```text
raw learned scores
-> mask structurally impossible candidates
-> existing ranking/downstream policy
```

This is preferred over adding another broad classifier.

### Stage H3 — ambiguity-only hybrid specialist

Only if H1/H2 leave a repeatable, causally understandable confusion group.

A specialist must:

- be narrowly scoped;
- have a predeclared trigger;
- operate only on surviving candidates;
- show benefit across multiple independent development corpora;
- not depend on CC-MAIN-2026-25.

### Stage H4 — cross-corpus robustness

Before any vNext production promotion:

- use multiple independent development corpora/crawls;
- pre-register the mechanism before evaluation;
- require directionally consistent gain;
- measure accuracy and runtime together;
- reject mechanisms whose gain is concentrated in one crawl or one tiny subgroup.

## Anti-overfitting rules

Do not:

- inspect R22 errors and create rules for those samples;
- retune S3, C, gate fraction, R12 threshold, or release-model gate thresholds from R22;
- add a new specialist because one crawl contains a frequent confusion;
- select encoding pairs by counting R22 failures;
- perform broad combinatorial parameter sweeps;
- promote a mechanism without a causal or specification-level explanation.

If R22 is ever used to choose a vNext mechanism, it ceases to be independent release evidence for that mechanism.

## Comparator learning rule

chardet 7 may be studied for architectural ideas, but CodecCat should not copy its full rule/model inventory.

Useful ideas to study:

- candidate elimination before expensive scoring;
- separation of structural evidence from statistical evidence;
- ambiguity-local work;
- heavy offline preparation with cheap runtime tables.

The research question is:

> Can CodecCat recover part of chardet's structural advantage while preserving a much smaller, faster, and more maintainable runtime?

## Promotion gate

A vNext mechanism may enter production consideration only if all are true:

1. explanation exists before benchmark selection;
2. it is not derived from R22 holdout errors;
3. it improves multiple independent development corpora;
4. runtime cost is small and explicitly measured;
5. implementation complexity remains bounded;
6. current frozen release behavior remains reproducible;
7. a new untouched holdout is reserved before vNext freeze.

Until then, R20/R21 remains the release candidate.
