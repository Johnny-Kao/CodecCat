# CodecCat Project Plan

Updated: 2026-10-06

## Product identity

CodecCat is a standalone character-encoding detection package.

It is **not**:
- a fork of `charset-normalizer`;
- a patch series for `charset-normalizer`;
- an upstream contribution to `chardet`;
- an OSS contribution workflow derived from another project.

CodecCat exists as an independent package with its own architecture, implementation, benchmark suite, release process, and maintenance model.

## Competitive references

CodecCat has exactly two primary benchmark references:

1. `charset-normalizer`
2. `chardet 7`

These projects are comparators, not upstreams.

Every external comparison must use the same held-out bytes, the same ground-truth normalization, and clearly separated accuracy and latency measurements.

## Product thesis

CodecCat should be competitive because it combines:

- **simple structure** — a small number of stable routing and scoring mechanisms;
- **low maintenance cost** — no unnecessary architectural layers or heuristic sprawl;
- **high speed** — avoid repeated byte scans, repeated feature extraction, repeated scorer execution, and generic abstractions in the hot path;
- **competitive accuracy** — preserve the experimentally validated 365/418 canonical result while improving generalization with more independent data;
- **retrainable parameters** — keep learned parameter state separable from the runtime architecture.

The target is not to reproduce the internal architecture of either comparator.

The target is to provide a smaller, easier-to-understand detector that is fast enough to be attractive in real Python workloads while remaining empirically competitive.

## Core optimization objective

```text
minimize expected compute
subject to bounded end-to-end detection error
and a maintainable runtime architecture
```

Runtime design target:

```text
cheap byte routing
+ compact fixed-shape feature extraction
+ small linear scoring kernels
+ narrowly justified calibration / specialist corrections
+ reusable inference state
```

## Current validated accuracy milestone

Canonical held-out evaluation:

- evaluated: **418**
- correct Top-1: **365**
- CodecCat Top-1: **87.3206%**
- charset-normalizer reference: **87.1429%**
- exact ranking identity maintained throughout the accepted performance optimization stages

This establishes that the current research architecture can match/slightly exceed the charset-normalizer reference on the canonical held-out benchmark.

It does **not** establish parity with chardet 7.

Historical held-out Common Crawl comparison before the final calibration work:

| Detector | Top-1 |
| --- | ---: |
| chardet 7 | **95.95%** |
| charset-normalizer | **87.14%** |
| early CodecCat model | **79.43%** |

Subsequent calibration and specialist work raised CodecCat to the current 365/418 canonical result.

## Current validated runtime architecture

The accepted performance work converged at **P15**.

Important accepted mechanisms accumulated across the research program include:

- one shared inference context instead of repeated scorer work;
- fused linear/scaler execution;
- indexed/precomputed downstream metadata;
- scalar small-class specialist kernels;
- cached fitted state for reproducible experimentation;
- route metadata and direct raw-array reuse;
- ndarray method-form ordering;
- preallocated feature output;
- native uint8 bigram wraparound;
- direct count assignment into float32 output plus in-place reciprocal normalization;
- algebraically simplified calibrator hot path with route-local static lookup;
- `np.add.reduceat` for fixed-range scalar aggregation;
- `np.dot(w, x) + b` for the small score kernel.

Final accepted stage:

- **P15**
- winner: `reduceat_dot`
- same-run improvement over the prior locked baseline: **1.04798x**
- runtime reduction: **~4.58%**

P16 re-profiled the optimized pipeline and found no further material low-risk full-pipeline winner.

Therefore routine micro-optimization is considered **converged** for the current Python/NumPy architecture.

## Research principles now frozen

1. Preserve the canonical output unless a future accuracy experiment explicitly changes the model.
2. Do not add heuristic complexity merely to win a small benchmark.
3. Prefer reuse and fixed-shape operations over generic repeated work.
4. Keep calibration/specialist rules only when supported by held-out evidence.
5. Accuracy claims must use held-out data.
6. Performance claims must use same-run controlled comparisons.
7. Comparator benchmarks must measure CodecCat, charset-normalizer, and chardet 7 on identical inputs.
8. Research harness code must remain separate from package runtime code.
9. The package implementation should be understandable without reading the historical experiment stack.
10. The runtime should remain small enough that future native acceleration can replace kernels without redesigning the public API.

## Package architecture target

Recommended production layout:

```text
src/codeccat/
    __init__.py
    api.py
    detector.py
    features.py
    scoring.py
    calibration.py
    model.py
    _types.py
```

The production package should not contain P-stage experiment classes or benchmark-specific names.

Public API should initially stay narrow. A minimal shape is:

```python
from codeccat import detect

result = detect(data)
```

The result should expose only stable user-facing information such as:

- encoding;
- confidence or score, once calibrated well enough to support a public semantic;
- optional ranked alternatives if the API is deliberately designed for them.

Do not expose internal route names, fold state, experimental calibrators, or training-only structures as public API.

## Productionization phases

### Phase A — research consolidation

Goal: translate the accepted research architecture into one readable runtime specification.

Deliverables:
- canonical algorithm description;
- accepted P1-P15 mechanism list;
- model/state requirements;
- public API boundary;
- explicit removal of failed/rejected experimental paths.

### Phase B — clean package implementation

Goal: implement CodecCat as an ordinary installable Python package.

Deliverables:
- `pyproject.toml`;
- `src/codeccat/` package;
- tests;
- typing;
- deterministic model/state loading;
- wheel and sdist build;
- no dependency on research scripts for ordinary inference.

### Phase C — package correctness validation

Required gates:
- canonical 365/418 result reproduced by the clean package;
- expected rank/output equivalence where applicable;
- unit tests for UTF-8/BOM/NUL/high-byte edge cases;
- empty and tiny inputs;
- deterministic results across repeated calls;
- supported Python-version matrix;
- clean wheel installation and import test.

### Phase D — formal comparator benchmark

Benchmark all three:

- CodecCat
- charset-normalizer
- chardet 7

Measure separately:

**Accuracy**
- Top-1;
- optional Top-3 / Top-5 where comparator APIs make the comparison meaningful;
- per-encoding-family breakdown;
- held-out WARC/domain grouping.

**Performance**
- latency per sample;
- throughput;
- memory/allocation footprint where practical;
- short / medium / large input buckets;
- ASCII/UTF-8-heavy and legacy-encoding workloads.

Do not mix accuracy and latency into a single headline number.

### Phase E — release readiness

CodecCat is release-ready when:

- package code is clean and independent of research harnesses;
- canonical accuracy is reproduced;
- benchmark methodology is reproducible;
- no material regression versus the locked research implementation;
- public README clearly explains where CodecCat is stronger or weaker than the two comparators;
- installation/build matrix passes.

## Competitive success criteria

CodecCat does not need to beat both comparators on every dimension.

A strong initial release is achieved if it demonstrates a clear Pareto position such as:

- accuracy near charset-normalizer with materially lower latency; or
- accuracy competitive with charset-normalizer while using a simpler runtime design; and
- a transparent gap to chardet 7 where chardet remains more accurate.

Longer-term target:

```text
charset-normalizer-class accuracy
+ substantially lower runtime cost
+ a much smaller and easier-to-maintain core
```

Then continue closing the remaining accuracy gap to chardet 7 without sacrificing that architectural simplicity.

## Historical research provenance

The original charset-detection research began in:

`Johnny-Kao/OSS-Engineering-Toolkit`
branch:
`research/charset-normalizer-e2e-baseline`

Historical files:
- `research/charset-detection/HANDOFF.md`
- `research/charset-detection/RESEARCH_PLAN.md`

Those files are historical provenance only.

As of 2026-10-06, **CodecCat is the source of truth for all future project planning and research state**.

The OSS Toolkit workflow does not govern this project.
