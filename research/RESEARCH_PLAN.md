# Charset Detection Replacement Research

Status: active experiment
Branch: `research/charset-normalizer-e2e-baseline`

## Goal

Design a charset-detection method/package that can replace `charset-normalizer`
and ultimately match or exceed `chardet 7` on correctness while improving
runtime simplicity, maintainability, and ideally speed/memory.

The target is not a small patch to either existing package.

## Core hypothesis

Real charset-detection calls occur in constrained scenarios. The first target
scenario is Network / HTTP fallback. Rather than solve arbitrary bytes with a
large hand-written heuristic tree, use:

```text
scene/context
→ cheap fixed byte features
→ scene/length-conditioned parameter table
→ array/LUT/linear scoring
→ small candidate set
→ second-stage verification
→ confidence/fallback
```

Long-term design target:

```text
stable runtime kernel
+ retrainable parameters
+ scene-conditioned weights
+ length-conditioned weights
```

Parameter updates should not require rewriting control logic.

## Baselines

Official `chardet` benchmark script was used on GitHub Actions (Ubuntu 24.04,
CPython 3.12) to avoid scoring-semantic drift.

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37178013550

Results on the 3,138-file chardet test suite:

| Detector | Encoding accuracy | Detection time | Median | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| chardet current main (mypyc + Cython) | 3130/3138 = 99.7% | 2.83 s | 0.31 ms | 2.04 ms | 7.46 ms |
| charset-normalizer 3.5.2 (compiled) | 2715/3138 = 86.5% | 3.31 s | 0.77 ms | 3.35 ms | 5.81 ms |

Note: chardet keeps a worse far CJK tail, while leading overall accuracy and
median/p95 latency.

## Target workload v0: web-origin

The first scenario is Network / HTTP fallback.

The initial web-origin subset is taken from provenance already present in the
public chardet test-data repository:

- CulturaX / mC4 / OSCAR Common Crawl samples;
- real RSS/Atom website XML;
- Wayback-derived samples;
- Chromium charset cases;
- Mozilla charset cases.

This produces 2,297 files.

Provisional length distribution:

| Bucket | Files |
| --- | ---: |
| 1–64 B | 1 |
| 65–256 B | 54 |
| 257–2,048 B | 529 |
| >2,048 B | 1,713 |

The bucket boundaries are experimental, not architecture commitments.

## Round 1 experiment: direct linear probe

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37178065082

Representation:

- 256 normalized byte unigram counts;
- 256 hashed byte-bigram counts;
- 6 scalar structural features;
- total: 518 features.

Classifier:

```text
x
→ standardization
→ linear W·x+b
→ rank encodings
```

Train/test split groups by source filename, so transcoded copies of the same
source text stay on one side of the split.

Results:

| Metric | Accuracy |
| --- | ---: |
| Top-1 | 66.67% |
| Top-3 | 94.95% |
| Top-5 | 97.47% |

Covered-class test (one test file had an unseen training class):

| Metric | Accuracy |
| --- | ---: |
| Top-1 | 66.84% |
| Top-3 | 95.19% |
| Top-5 | 97.72% |

Interpretation:

This is not competitive as a final detector. It is strong evidence that a very
cheap first-stage candidate reducer may be viable: a crude fixed-array model
already retains the true encoding in a five-candidate set for ~97.5% of held-out
web-origin cases without charset-specific heuristic logic.

## Research sequence

### R0 — competitor baseline
Done.

### R1 — information / separability
Active.

1. Sparse exact byte n-gram feature pool.
2. Offline feature selection.
3. Sweep retained features.
4. Measure Top-1 / Top-3 / Top-5 with group-held-out data.
5. Test length-conditioned parameter tables.
6. Stop if candidate recall cannot approach/exceed the chardet target.

### R2 — minimum sufficient features
Only if R1 succeeds.

Reduce selected feature count while measuring the accuracy knee:
`4096 → 2048 → 1024 → 512 → 256 → ...`

### R3 — execution kernel
Only after the mathematical representation is proven.

Compare:

- dense array / dot product;
- integer accumulator;
- LUT;
- SIMD;
- sparse selected n-gram lookup.

Do not assume a general matrix multiply is fastest.

## Acceptance target

The final method should aim for:

- end-to-end correctness at least at chardet-7 level on the target workload;
- equal or lower runtime than chardet on the target workload;
- bounded or improved memory;
- 2–3 logical runtime stages;
- stable code with retrainable parameter tables;
- explicit confidence/fallback semantics.

## Benchmark rules

- Formal runtime comparisons run on GitHub Actions, not the contributor's local
  M5 environment.
- Same runner type, Python version, dataset, scoring semantics, and repeated
  isolated processes for A/B.
- Local-machine measurements are exploratory only.
- Avoid GitHub Actions artifacts while account artifact quota is constrained;
  persist durable conclusions/results as repository text instead.


## Round 1 experiment 2: exact sparse n-gram sweep

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37178171703

Method:

- raw feature space: 256 exact unigrams + 65,536 exact byte bigrams;
- offline chi-square feature selection;
- retained 256 / 512 / 1,024 / 2,048 / 4,096 features;
- LinearSVC scoring;
- same group-held-out split as experiment 1.

Result:

| Selected features | Top-1 | Top-3 | Top-5 |
| ---: | ---: | ---: | ---: |
| 256 | 48.74% | 83.33% | 84.34% |
| 512 | 49.24% | 83.84% | 85.10% |
| 1,024 | 49.49% | 84.34% | 87.12% |
| 2,048 | 50.00% | 84.09% | 88.13% |
| 4,096 | 50.25% | 84.34% | 88.38% |

Interpretation:

Naively retaining more exact n-grams did not improve over the low-dimensional
hashed representation. This is evidence against the assumption that a larger,
more literal n-gram table is automatically better. The smoother hashed
representation remains the stronger first-stage representation so far.

Next experiment: keep the 518-dimensional representation fixed and test
length-conditioned parameter tables. This isolates the conditioning hypothesis
without adding model complexity.


## Round 1 experiment 3: positional / structural ablation

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37179530304

Using the 2,297-file web-origin set, the same linear scorer was tested with:

- B = 518-dimensional base byte representation;
- P = 64 positional coarse-byte features (first 64 B, head, middle, tail);
- S = 33 structural/repetition features;
- L = 5 length features.

Lenient Top-k results:

| Variant | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: |
| B | 78.28% | 96.21% | 97.73% |
| B+P | 77.53% | 96.46% | 98.23% |
| B+S | 78.28% | 96.72% | 97.98% |
| B+P+S | 77.27% | 96.72% | 98.23% |
| B+P+S+L | 77.02% | 96.97% | 98.23% |

Position carries measurable candidate-recall information. Structural features help
Top-3 somewhat. Adding all families does not improve Top-5 beyond B+P.

## Top-5 miss analysis and workload correction

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37179620741

B+P leaves 7 Top-5 misses in 396 held-out files. The misses include:

- ASCII Chromium no-encoding-specified case;
- Big5 RSS/XML;
- CP437 CulturaX;
- CP860 CulturaX;
- CP949 RSS/XML;
- two UTF-8 CulturaX Welsh samples.

This exposed a workload-definition problem: CulturaX text is web-derived, but
many test-data files are deliberately transcoded offline into legacy encodings.
Those bytes are not evidence for the real Network/HTTP encoding prior.

Therefore distinguish:

1. **web-origin text** — source content came from the web, possibly transcoded;
2. **strict network bytes** — bytes plausibly observed in an actual
   network/browser charset-fallback setting.

This distinction is now part of the research methodology.

## Round 1 experiment 4: strict-network subset

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37179713511

Initial strict-network selector excludes CulturaX transcoded copies and keeps
real RSS/Atom XML, Wayback-derived samples, Chromium cases, and Mozilla cases.

Dataset:

- 351 files total;
- 271 train;
- 80 held-out test.

Results:

| Variant | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: |
| B | 93.75% | 97.50% | 98.75% |
| B+P | 92.50% | 97.50% | 98.75% |
| B+S | 92.50% | 97.50% | 98.75% |
| B+P+S | 92.50% | 97.50% | 98.75% |
| B+P+S+L | 92.50% | 96.25% | 98.75% |

Interpretation:

The largest gain came from defining the scene correctly, not from adding model
complexity. Base-model Top-1 rose from 78.28% on web-origin-plus-transcoded data
to 93.75% on strict-network bytes.

This strongly supports the scene-conditioned-detector hypothesis, but the
strict-network sample is currently too small for a final accuracy claim.
The next priority is increasing real/wild network-byte coverage before
optimizing the runtime kernel further.


## Round 1 experiment 5: 5-fold stability and byte-cost frontier

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37180168296

The strict-network subset currently contains:

- 351 files;
- 29 encodings;
- provenance: 314 RSS/Atom XML, 15 Chromium, 11 Mozilla, 11 Wayback;
- median file size 10,056 B;
- p95 54,145 B;
- max 596,838 B.

This confirms that the current dataset is still provenance-imbalanced and must
be expanded before any headline accuracy claim.

Five-fold grouped cross-validation was used to reduce single-split noise.
The same 518-dimensional base representation was evaluated under multiple
evidence budgets:

| Evidence policy | Mean bytes examined | Lenient Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| Full file | 16,904 B | 89.81% | 96.63% | 97.25% |
| First 8 KiB | 6,320 B | 88.18% | 95.38% | 96.34% |
| First 2 KiB | 1,884 B | 84.23% | 91.71% | 94.41% |
| First 512 B | 507 B | 84.64% | 91.58% | 95.57% |
| Head + middle + tail (768 B max) | 749 B | 88.30% | 94.62% | 97.12% |

Head/middle/tail sampling is the strongest cost result so far. Relative to the
full-file representation, it examines about 749 B instead of 16.9 KiB on
average (roughly 95.6% fewer bytes by ratio of means), while losing only about
0.13 percentage points of lenient Top-5 recall.

The Python prototype's feature extraction averaged about 0.030 ms/file for
H/M/T 768 B versus about 0.103 ms/file for full-file features. These timings
are exploratory only; the production target remains a native array/LUT kernel.

Five-fold variance is still material because the dataset is small. Full-file
Top-1 ranged roughly 84.1%-95.2%, and H/M/T Top-1 roughly 82.1%-95.0%.
This reinforces the need for a larger and more diverse real-network corpus.

### Current interpretation

The evidence policy is now part of the model architecture:

```text
cheap bounded evidence
→ score / confidence
→ accept if reliable
→ acquire more evidence or invoke verifier only when necessary
```

This is preferable to always scanning the full input.

## Benchmark methodology adopted from chardet

Future evaluations should preserve these ideas from chardet's benchmark suite:

- publish lenient and strict accuracy together;
- report latency distribution (median, p90, p95, p99), not mean alone;
- split script families where tails differ materially (especially CJK);
- compare pure/native implementations separately;
- verify rankings across Python versions / runner architectures;
- separate one-time model/import memory from incremental per-detection memory;
- use interleaved A/B rounds for timing ratios;
- measure large-input behavior and bytes/evidence examined;
- fingerprint or otherwise decontaminate train/test sources.

## Academic connections that directly inform the design

### Small n-gram profiles
Cavnar & Trenkle showed that compact n-gram frequency profiles can classify
language robustly. This supports searching for a minimum sufficient
representation rather than maximizing feature count.

### Length-dependent statistical order
Dunning's statistical language-identification work found that the useful model
order changes with test-string length. For this project, length should therefore
influence evidence policy / calibration before it is allowed to multiply the
number of independent weight matrices.

### Hashed byte n-grams + linear scoring
byteSteady uses byte-level n-grams, hashing, averaged representations and a
linear classifier. This is close to the strongest representation found in our
early experiments and supports keeping runtime math simple and cache-friendly.

### Cost-sensitive feature acquisition
Cost-sensitive classification literature treats features as having test-time
acquisition cost and asks whether more evidence is worth buying. This maps
directly to prefix/full-file scans, structural verification, and decode checks.

### Selective classification / reject option
Selective-classification work evaluates risk versus coverage: a cheap classifier
answers only when confident and defers uncertain instances. This suggests
evaluating a cascade by risk-coverage and expected compute, not accuracy alone.

## Proposed primary objective

Do not collapse evaluation prematurely into one arbitrary scalar.

Primary reporting should use a Pareto frontier across:

- correctness / conditional risk;
- coverage handled by the cheap stage;
- bytes examined;
- p50 / p95 / p99 latency;
- steady-state memory;
- fallback rate.

A useful constrained objective is:

```text
minimize expected compute cost
subject to end-to-end error <= target
```

or, for a selective first stage:

```text
maximize cheap-stage coverage
subject to accepted-stage error <= epsilon
```

This preserves the project's intended win condition: ideally dominate on
accuracy, speed and memory, but still count a substantially cheaper detector at
the same bounded error as a meaningful win.


## Round 1 experiment 6: selective cascade and confidence calibration

Runs:

- Raw selective cascade:
  https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37180290227
- Confidence estimator ablation:
  https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37180382286
- Normalized-confidence cascade:
  https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37180452324

A cheap H/M/T-768 stage was evaluated as a selective classifier before chardet.

### Raw margin is not a reliable gate

Using the unnormalized top1-top2 score margin, even the highest-margin 20% of
samples were only 97.1% correct. A cascade that accepted those samples degraded
end-to-end accuracy below the chardet baseline.

This confirms that classifier score magnitude is not automatically a calibrated
confidence measure.

### Normalized logits materially improve risk-coverage

An L2-normalized top-logit confidence score produced the following OOF
accepted-stage accuracy on the 351-file strict-network set:

| Cheap-stage coverage | Accepted accuracy |
| ---: | ---: |
| 10% | 100.0% |
| 20% | 100.0% |
| 30% | 100.0% |
| 40% | 98.57% |
| 50% | 98.30% |

This mirrors recent selective-classification findings that post-hoc logit
normalization can repair confidence ranking without changing the base classifier.

The sample is too small to interpret 30% / zero observed errors as a guaranteed
risk bound; it is a direction-finding result only.

### Cascade result

On this subset, chardet scored 350/351 = 99.715%.

With normalized confidence:

| Cheap-stage coverage | Cheap accepted accuracy | Cascade accuracy | Fraction of chardet fallback time retained |
| ---: | ---: | ---: | ---: |
| 10% | 100.0% | 99.715% | 96.65% |
| 20% | 100.0% | 99.715% | 93.41% |
| 30% | 100.0% | 99.715% | 90.43% |
| 40% | 98.57% | 99.15% | 86.12% |

The 30% gate preserves the observed chardet accuracy, but saves only about 9.6%
of measured chardet fallback time. The accepted high-confidence cases are
disproportionately cheap for chardet already.

This changes the optimization target.

### Revised cascade objective

Do not maximize coverage alone.

Prefer:

```text
maximize safely avoided fallback compute
subject to accepted-stage risk <= epsilon
```

The gate should eventually combine:

- calibrated correctness confidence;
- predicted fallback cost (length, script family, structural ambiguity, candidate
  family, or a learned cheap cost predictor).

Conceptually:

```text
gate value = reliability × avoidable downstream cost
```

This is a direct test-time cost-sensitive classification problem.

## Dataset expansion: real WARC network bytes

The next corpus should be built from real HTTP response bytes rather than
offline-transcoded web text.

Preferred source: Common Crawl WARC response records.

Two evaluation distributions should be maintained:

1. **natural-frequency** — preserves realistic web charset prevalence and is
   used for expected latency/cost;
2. **balanced-challenge** — stratified across legacy encodings / script families
   and domains so UTF-8 prevalence cannot hide weak legacy behavior.

High-confidence inclusion rules should require independent encoding evidence
where possible (e.g. compatible HTTP charset + HTML/XML declaration or BOM),
successful strict decoding, domain caps, source fingerprinting and train/test
decontamination.

The current 351-file strict-network set is 314 RSS/Atom XML, 15 Chromium,
11 Mozilla and 11 Wayback records, so it is not sufficiently diverse for a
headline accuracy claim.


## Stable HTTP response structure tree

The runtime taxonomy should be based on **stable byte structure**, not product or
MIME names. Different response names that produce the same structural evidence
should share one branch and the same runtime kernel.

Length, position, provenance and encoding family are orthogonal conditioning
signals. They should not multiply the number of structural branches.

### Stage 0 — should charset detection run?

```text
HTTP response
│
├─ trusted charset / BOM gives a decisive answer
│    └─ return normalized encoding
│
├─ clearly non-text / binary payload
│    └─ do not run charset detector
│
└─ textual or unknown body with insufficient charset metadata
     └─ enter structural selector
```

This stage should consume metadata already available to the caller whenever
possible. It should not parse the whole body just to decide whether detection is
needed.

### Stage 1 — cheap exact byte fast paths

Before statistical classification, check only evidence that can produce a
high-confidence answer cheaply:

```text
textual / unknown bytes
│
├─ BOM / strong UTF-16/32 pattern
│    └─ Unicode exact path
│
├─ all-ASCII window and policy permits ASCII/superset answer
│    └─ ASCII/superset fast path
│
├─ structurally valid UTF-8 with sufficient evidence
│    └─ UTF-8 candidate / exact path according to policy
│
└─ unresolved
     └─ structural family selector
```

Whether UTF-8 validity is definitive or only a strong candidate remains an
empirical policy decision; do not hard-code it before validation.

### Stage 2 — structural family selector

Use a small sequence of binary tests. The order is deliberately from cheap and
highly distinctive to more generic.

```text
unresolved textual body
│
├─ markup-like?
│    │
│    ├─ YES → MARKUP family
│    │          HTML
│    │          XML
│    │          RSS / Atom
│    │          SVG / XHTML
│    │          server-generated HTML error pages
│    │
│    └─ NO
│
├─ repeated record / line structure?
│    │
│    ├─ YES → RECORD family
│    │          CSV / TSV
│    │          subtitle / captions
│    │          M3U / manifests
│    │          line-oriented exports
│    │          many log/config-like responses
│    │
│    └─ NO
│
├─ punctuation-dense code / key-value structure?
│    │
│    ├─ YES → CODE family
│    │          JSON
│    │          JavaScript
│    │          CSS
│    │          source/config-like API bodies
│    │
│    └─ NO
│
└─ TEXT family
       plain text
       prose
       unknown textual body
       short server messages
```

The selector does **not** need to identify the exact content type. It only needs
to choose the structural parameter family that changes charset evidence or
evidence-acquisition policy.

### Why these families are intentionally merged

| Named formats | Runtime structural family | Shared evidence |
| --- | --- | --- |
| HTML / XML / RSS / Atom / XHTML / SVG | MARKUP | angle brackets, tags, declarations, attribute syntax, strong positional metadata |
| CSV / TSV / subtitle / manifest / many logs | RECORD | repeated lines, separators, recurring field/record shapes |
| JSON / JS / CSS / config-like API output | CODE | ASCII punctuation density, braces/brackets/key-value syntax, code-like repetition |
| TXT / prose / short messages / unknown text | TEXT | little reliable syntax; byte/statistical evidence matters most |

Error pages are not a separate family: they become MARKUP or TEXT depending on
their actual body. Very short responses are not a separate family either:
length modifies the evidence policy and confidence threshold.

### Orthogonal conditioning axes

After structural selection, the scorer can condition on a small set of
orthogonal signals without creating more code branches:

```text
structural family
× length regime
× byte-position evidence
× caller metadata / provenance
× candidate encoding family
```

Conceptually:

```text
profile = parameters[family][length_bucket]
score   = fixed_kernel(features, profile)
```

The implementation should prefer one stable kernel with different parameter
tables over separate algorithms for each family.

### Target binary selector

The desired selector is approximately:

```text
needs detector?
  ↓ yes
exact Unicode / ASCII fast path?
  ↓ no
markup-like?
  ├─ yes → MARKUP
  └─ no
      ↓
record-like?
  ├─ yes → RECORD
  └─ no
      ↓
code-like?
  ├─ yes → CODE
  └─ no  → TEXT
```

Each decision should be implementable from bounded bytes using counters,
small lookup tables, fixed delimiters and simple threshold comparisons. Avoid
general parsers, regex engines, object-heavy Python structures, or full decoding
inside the selector.

### Runtime architecture after the tree stabilizes

```text
Python API
   ↓
thin validation / metadata wrapper
   ↓
native selector
   ├─ exact fast paths
   └─ MARKUP / RECORD / CODE / TEXT
            ↓
bounded evidence sampler
            ↓
fixed feature extraction
            ↓
array / LUT / integer scoring kernel
            ↓
calibrated confidence + expected fallback cost
            ↓
accept OR acquire more evidence / verifier
```

The structural tree is intentionally simple enough to compile into a low-level
implementation later. Candidate implementation paths include Cython/C-extension,
mypyc-compatible Python, Rust/C, or another native backend, but the language
choice should be made only after the stable algorithm and memory layout are
measured.

The maintenance target is therefore:

```text
stable selector + stable native kernel + retrainable parameter tables
```

rather than a growing collection of format-specific heuristics.


## Structural selector status: empirical validation required

The previously sketched MARKUP / RECORD / CODE / TEXT tree is **not considered
validated architecture**. It is a hypothesis tree only. Structural gates must be
accepted or rejected from measured byte-feature separability.

### Current feature-validation corpus

Using chardet/test-data and grouping files by extension as a proxy for content
structure:

- MARKUP: 723 files (.html/.xml)
- TEXT: 2,383 files (.txt/.po/.md/.rst)
- CODE: 32 files (.json/.py/.hpp)
- RECORD: 13 files (.csv/.srt/.m3u)

All experiments use only a bounded 4 KiB prefix and cheap counters/ratios.
Same filenames across encoding directories are grouped into the same CV fold to
reduce transcode leakage.

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37184855368

A 4-way shallow tree reaches only about 81.3% balanced accuracy at depth 4.
Therefore the four-family selector is not validated as a direct classifier.

### Binary-gate validation

Runs:

- Initial gates:
  https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37184935807
- Merged structured-nonmarkup test:
  https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37185003169

#### G1 — MARKUP vs non-MARKUP

This gate is strongly supported by current data.

At tree depth 2:

- balanced accuracy: 97.73%
- MARKUP recall: 99.59%
- specificity: 95.88%
- 3 false negatives / 100 false positives over 3,151 files

The dominant feature is simply '<' frequency, with slash / repeated-line
statistics providing small corrections.

This gate is a plausible fast selector candidate, subject to validation on
real WARC/network bytes.

#### G2 — structured non-MARKUP vs TEXT

RECORD and CODE were merged because their byte structures may be closer to each
other than to prose/text.

At depth 2:

- positive set: 45 files (RECORD + CODE)
- balanced accuracy: 89.18%
- recall: 86.67%
- specificity: 91.69%
- precision: only 16.46%

The gate has useful signal (line-length repetition, comma-bearing lines, length)
but is **not yet production-ready**. The low precision is strongly affected by
the very small positive sample and by structure-like plain-text files.

#### G3 — RECORD vs CODE

On only 45 total positive-domain files (13 RECORD, 32 CODE), the best observed
balanced accuracy is about 82.93%.

A simple quote-density split dominates. This result is too data-starved to
justify a runtime branch.

### Current structural conclusion

Only the first split is empirically supported:

```text
text-like input
│
├─ MARKUP-like      ← currently supported
└─ non-MARKUP       ← do NOT subdivide yet
```

The next split should remain an experimental question:

```text
non-MARKUP
├─ structured non-markup?
└─ general text?
```

Do not hard-code RECORD / CODE / TEXT branches until the corpus contains
substantially more real-network examples of JSON/API, CSV/tabular, subtitles,
manifests, source-like bodies, and plain text.

The selector design rule is now:

> **feature evidence first, tree second.**

Each proposed binary gate must show stable separability under grouped
cross-validation before it becomes part of the runtime architecture.


## Cost-aware routing-tree experiment v0

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37187015497

### Corpus provenance

The first routing-tree experiment combines the two existing public benchmark
corpora used by the competitors:

1. `chardet/test-data`;
2. `Ousret/char-dataset` (charset-normalizer's challenge corpus).

Raw labeled files included by the loader:

- chardet/test-data: 3,130 textual labeled files;
- char-dataset: 469 textual labeled files.

The corpora overlap heavily. Files are SHA-256 deduplicated before training.
After deduplication the experiment contains 2,952 unique byte sequences covering
90 normalized encoding labels. Duplicate corpus entries with conflicting labels
are excluded rather than silently reconciled.

Cross-validation groups samples by source basename so the same source file
transcoded to different encodings is less likely to cross train/test folds.

### Training objective

A constrained greedy binary routing tree is trained with:

- maximum depth: 4;
- maximum leaves: 6;
- minimum leaf size: 8.

Candidate split utility is initially:

```text
encoding entropy reduction / relative feature acquisition cost
```

This is a routing experiment, not a final charset classifier.

### Result

With bytes/context features available, the first v0 tree learns that extremely
cheap information dominates early routing.

**Bytes-only**:
- dominant early feature: body length;
- mean training encoding-entropy reduction: ~10.5%.

**Free context + bytes**:
- dominant early features: known text-format hint + body length;
- mean training encoding-entropy reduction: ~15.7%.

This supports the hypothesis that free caller metadata and very cheap features
should be consumed before expensive byte statistics.

The leaf-prior Top-k scores are intentionally not treated as detector accuracy:
a routing leaf is supposed to own a separately trained parameter profile, not
return its most common encoding.

### Method correction for v1

The v0 cost model prices each byte statistic separately. That is not faithful to
a native implementation: one bounded byte scan can co-acquire many counters.

The next training objective should therefore use **feature bundles**:

```text
free metadata bundle
→ length / size bundle
→ bounded byte-scan bundle
→ optional positional / structural bundle
→ optional expensive verifier
```

A bundle's acquisition cost is paid once even if several derived features are
used later on the path.

The decisive next A/B is:

```text
universal W,b
vs
routing tree + per-leaf W_leaf,b_leaf
```

under the same grouped out-of-fold split.


### Routed profile A/B result

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37187103829

Each routing leaf was given its own 518-dimensional H/M/T-768 linear scorer and
compared out-of-fold with one universal scorer trained on the same feature
representation.

| Model | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: |
| Universal scorer | 62.09% | 86.08% | 90.55% |
| Bytes-only routing + leaf profiles | 58.60% | 81.06% | 85.50% |
| Context routing + leaf profiles | 61.45% | 82.76% | 87.74% |

The v0 routing tree therefore does **not** improve the detector. Entropy
reduction alone is not a sufficient routing objective: splitting reduces the
training data available to each leaf, and the resulting specialization loss can
outweigh the entropy gain.

This negative result changes the training objective.

### Routing-tree v1 objective

A candidate split should be accepted only when its child models improve
out-of-fold detector loss enough to justify both:

1. feature-acquisition cost;
2. specialization / reduced-sample penalty.

Conceptually:

```text
split value =
(parent validation loss - weighted child validation loss)
/
incremental acquisition cost
```

subject to minimum child sample and class coverage constraints.

The selector must therefore be trained jointly with the downstream scorer,
rather than learned from encoding entropy in isolation.


## Downstream-loss-aware selector tournament and convergence

### Tournament harness

The first downstream-loss-aware routing implementation was too expensive because
candidate splits repeatedly retrained child logistic scorers. A single run hit the
Actions timeout / cancellation boundary.

The harness was then changed to:

- one-process execution;
- coarse-to-fine split search;
- cheap proxy ranking over all candidate thresholds;
- exact downstream scorer evaluation only for the top proxy candidates;
- global child-loss caching by partition + regularization;
- shared corpus / fold / feature matrices;
- workflow concurrency to prevent duplicate runs.

This reduced the 6-method Round-1 screen to about 139 seconds.

Round-1 run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37201083741

Round-1 survivors:
- D
- A
- B

Universal 3-fold baseline:
- Top-1: 64.997%
- Top-3: 88.854%
- Top-5: 93.112%

Best Round-1 method D:
- Top-1: 72.010%
- Top-3: 91.672%
- Top-5: 95.679%

This was the first clear held-out evidence that routing + per-leaf profiles can
beat the same universal 518-dimensional scorer.

### Round 2: D / A / B × 3 configurations

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37205949226

Winner:
- method D
- C = 0.5
- lambda_cost = 0
- max_leaves = 5

Metrics:
- Top-1: 72.135%
- Top-3: 92.235%
- Top-5: 95.304%
- mean routing depth: 2.22
- p95 routing depth: 4
- median leaf size: 567
- median leaf encoding classes: 21.5

Against the universal scorer from the same run:
- Top-1: +7.01 pp
- Top-3: +3.19 pp
- Top-5: +2.25 pp

Interpretation:

The winning structure uses relatively broad leaves. This supports routing into a
small number of mechanism/profile groups, not aggressive fragmentation into many
tiny experts.

### Winner split stability

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37207402007

The first two learned decisions were identical across all three grouped folds:

1. root: `utf8_valid_4k > 0.5`
2. next: `nul_ratio > 0`

Both feature choice and threshold were stable:
- `utf8_valid_4k`: threshold 0.5 in 3/3 folds
- `nul_ratio`: threshold 0 in 3/3 folds

These two decisions separate broad encoding mechanisms:

- UTF-8-valid / ASCII / UTF-7-like cases;
- NUL-bearing UTF-16 / UTF-32 families;
- a residual non-UTF8, non-NUL legacy/multibyte region.

This is substantially stronger evidence than the earlier human-sketched
MARKUP/RECORD/CODE/TEXT taxonomy. The current selector should therefore be
thought of as **encoding-mechanism routing**, not format classification.

### Residual discovery

Residual definition:

```text
utf8_valid_4k <= 0.5
AND
nul_ratio == 0
```

Residual size:
- 1,346 samples
- 77 encodings

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37213243765

A single scalar third split was not stable:
- fold 0 root: `high_byte_ratio`
- fold 1 root: `known_text_ext`
- fold 2: no positive downstream-loss split

Therefore no fixed third scalar selector was accepted at this stage.

### Residual convergence batch

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37214176887

Compared:
- shared residual scorer;
- multiple tiny 2–4 feature selectors;
- selector depths 1 and 2.

Best result:
- M2, depth 2
- candidate features: `high_byte_ratio`, `ascii_only_4k`
- actual learned feature use: **high_byte_ratio only**
- residual Top-1: 70.05%
- residual Top-3: 84.87%
- residual Top-5: 89.56%

Shared residual scorer:
- Top-1: 68.23%
- Top-3: 84.57%
- Top-5: 88.80%

Delta of the best tiny selector:
- Top-1: +1.82 pp
- Top-3: +0.30 pp
- Top-5: +0.76 pp

Adding comma ratio, line-length CV, markup punctuation, newline ratio and similar
features did not produce a stable improvement and often reduced performance.

Current evidence therefore favors:

```text
bounded sample
→ UTF-8 validity
→ NUL presence
→ high-byte-ratio refinement
→ small leaf scorer
```

rather than a larger multi-feature structural selector.

### High-byte-ratio threshold stability

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37214743877

A dedicated two-threshold grid was run inside the residual region.

Per-fold best pairs:

| Fold | t1 | t2 | Top-1 | Top-3 | Top-5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.02 | 0.25 | 72.28% | 86.14% | 89.60% |
| 1 | 0.03 | 0.60 | 72.11% | 91.24% | 94.02% |
| 2 | 0.03 | 0.60 | 74.52% | 87.98% | 91.83% |

Median fold-best pair:
- t1 = 0.03
- t2 = 0.60

Best fixed pair by mean cross-fold score:
- **t1 = 0.03**
- **t2 = 0.25**

Fixed-pair residual performance by fold:

| Fold | Top-1 | Top-3 | Top-5 |
| ---: | ---: | ---: | ---: |
| 0 | 71.78% | 85.15% | 90.59% |
| 1 | 71.31% | 89.64% | 93.63% |
| 2 | 74.52% | 87.02% | 91.35% |

Interpretation:

- the **lower high-byte-ratio cut is stable around 0.02–0.03**;
- the **upper cut is not yet stable** (0.25 vs 0.60 across folds);
- `t1=0.03, t2=0.25` is the most robust fixed pair on the current corpus;
- this two-threshold test creates three residual buckets and is not identical to
  the prior depth-2 tree with four leaves, so it should be treated as a
  threshold-stability probe rather than final architecture proof.

### Current selector hypothesis

The empirically supported part is now:

```text
S1: utf8_valid_4k > 0.5
S2: nul_ratio > 0
S3: high_byte_ratio around 0.02–0.03
S4: possible upper high_byte_ratio cut, not yet stable
```

Do not freeze S4 yet.

The next high-value work is external validation and/or more robust threshold
estimation, not adding more arbitrary structural features.


## S4 necessity / stability validation

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37250023917

The upper high-byte-ratio split was tested as an architecture question rather
than another threshold search.

Fixed comparison:
- residual region: `utf8_valid_4k <= 0.5 AND nul_ratio == 0`
- S3 lower cut fixed at `high_byte_ratio = 0.03`
- S3-only: two residual buckets
- S3+S4: three residual buckets with upper cut fixed at 0.25 or 0.60
- 5 grouped out-of-fold folds
- paired bootstrap over source-name groups, 3,000 draws

Pooled metrics:

| Config | Top-1 | Top-3 | Top-5 | Composite |
| --- | ---: | ---: | ---: | ---: |
| S3 only | 64.12% | 82.76% | 87.96% | 5.0996 |
| S3 + 0.25 | 64.93% | 83.28% | 88.71% | 5.1501 |
| S3 + 0.60 | 63.82% | 83.36% | 88.86% | 5.1085 |

For S4=0.25, composite gain was positive in 4/5 folds, but the paired
group-bootstrap 95% interval still crossed zero: `[-0.0140, +0.1065]`.
The pooled Top-1/3/5 gains were +0.82 / +0.52 / +0.74 percentage points.

For S4=0.60, composite gain was positive in only 2/5 folds; pooled Top-1 fell
by 0.30 percentage points and the composite bootstrap interval was
`[-0.0432, +0.0572]`.

### Decision

Do **not** freeze S4 on the current evidence.

The minimal selector candidate is now:

```text
S1: utf8_valid_4k > 0.5
S2: nul_ratio > 0
S3: high_byte_ratio low cut around 0.02–0.03
then: one residual scorer for the remaining high-byte region
```

The 0.25 upper cut remains a research signal, not runtime architecture.
It should be reconsidered only if independent external/WARC data reproduces the
gain with a confidence interval above zero and consistent fold-level sign.

Next high-value work is external validation of S1/S2/S3, not further in-corpus
S4 threshold tuning.


## External Common Crawl validation — first real-network result

Natural-frequency fast run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37251360584

Route-balanced residual run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37251555242

Source:
- Common Crawl `CC-MAIN-2026-39`
- real `WARC-Type: response` records
- deterministic WARC sampling
- only high-confidence charset labels retained
- Tier A: agreeing independent signals or BOM
- Tier B: explicit HTTP charset + strict decode with no conflicting body declaration

The first natural-frequency sample accepted 300 responses, but 283/300 were
UTF-8-valid and only 17 reached the residual region. This confirmed that a
naive web sample is too UTF-8-heavy to validate S3.

The route-balanced rerun capped the UTF-8-valid route and collected:
- 180 accepted responses
- 168 Tier A / 12 Tier B
- 120 residual non-UTF8/non-NUL cases
- 60 UTF-8-valid cases
- labels included cp1251, gb18030, Big5, ISO-8859-1/2, EUC-JP/KR, Shift-JIS,
  UTF-8 and UTF-8-SIG

External fixed-routing results:

| Selector | Top-1 | Top-3 | Top-5 | Composite |
| --- | ---: | ---: | ---: | ---: |
| no S3 | 31.11% | 45.00% | 56.67% | 2.7111 |
| S3 = 0.02 | 31.67% | 47.22% | 57.78% | 2.7889 |
| S3 = 0.03 | 31.11% | 46.67% | 57.78% | 2.7556 |

S3=0.02 vs no S3:
- Top-1: +0.56 pp
- Top-3: +2.22 pp
- Top-5: +1.11 pp
- composite: +0.0778

S3=0.03 vs no S3:
- Top-1: +0.00 pp
- Top-3: +1.67 pp
- Top-5: +1.11 pp
- composite: +0.0444

Interpretation:
- S3 survives the first external real-network challenge as a useful split.
- External evidence shifts the preferred lower cut toward **0.02**, not 0.03.
- Do not freeze the exact threshold yet; current evidence supports a
  `~0.02–0.03` band with 0.02 currently stronger externally.
- S4 remains rejected.
- The much larger problem is now scorer generalization: overall external Top-1
  is only ~31%, despite routing improving relative performance.
- Therefore the next bottleneck is **profile/scorer domain shift**, not adding
  another selector.

Current working runtime hypothesis:

```text
S1 UTF-8 validity
→ S2 NUL presence
→ S3 high-byte-ratio low cut (~0.02–0.03; external lean = 0.02)
→ scorer/profile
```

Next priority:
1. diagnose external scorer failure by encoding family / bucket;
2. determine whether retraining the same 518-dim H/M/T scorer on real-network
   bytes fixes the gap;
3. only then freeze selector threshold and optimize the native kernel.


## External scorer domain-shift A/B — decisive result

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37252723794

Question:
> Is the external accuracy collapse mainly a training-distribution mismatch, or
> does the 518-dimensional H/M/T scorer representation itself fail on real
> network bytes?

Setup:
- selector fixed: S1 UTF-8 validity, S2 NUL presence, S3 high-byte-ratio = 0.02
- same 518-dimensional H/M/T scorer representation
- same logistic scorer family
- Common Crawl `CC-MAIN-2026-39`
- 260 high-confidence real HTTP responses
- 2-fold domain holdout by WARC path
- compare:
  - legacy-only scorer training
  - legacy + external scorer training using only the opposite WARC fold

External sample:
- 260 responses
- 241 Tier A / 19 Tier B
- routes: 149 RH, 31 RL, 80 U
- labels include cp1251, Big5, GB18030, ISO-8859-1/2, EUC-JP/KR,
  Shift-JIS, KOI8-R, UTF-8 and UTF-8-SIG

Pooled held-out results:

| Training | Top-1 | Top-3 | Top-5 | Composite |
| --- | ---: | ---: | ---: | ---: |
| legacy only | 32.17% | 46.90% | 56.59% | 2.7907 |
| legacy + external | 66.67% | 85.66% | 90.31% | 5.2829 |

Delta from adding real-network training bytes:
- Top-1: **+34.50 pp**
- Top-3: **+38.76 pp**
- Top-5: **+33.72 pp**
- composite: **+2.4922**

Both WARC-held-out directions improved sharply:
- fold 0 Top-1: 16.98% → 66.04%
- fold 1 Top-1: 36.10% → 66.83%

Interpretation:
- The external failure is primarily **training-distribution mismatch**.
- The current 518-dimensional H/M/T representation retains strong useful
  information on real-network bytes.
- Do **not** redesign the representation yet.
- Do **not** add selector complexity.
- S3=0.02 remains the current external-leading lower cut.
- The next bottleneck is building enough diverse real-network training data to
  learn stable parameter profiles without same-domain overfitting.

Next priority:
1. expand real-network corpus across more WARC files / domains / legacy encodings;
2. keep domain-held-out evaluation;
3. retrain the same scorer/profile architecture;
4. measure the learning curve to see how many real-network samples are needed
   before Top-1/Top-3/Top-5 plateau;
5. only after that compare directly against chardet 7 and charset-normalizer
   on the same held-out real-network corpus.


## External scorer learning curve — no plateau yet

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37253597982

Setup:
- Common Crawl `CC-MAIN-2026-39`
- 420 high-confidence real HTTP responses
- 43 WARC files
- 4-fold WARC-path holdout
- selector fixed: S1 UTF-8 validity → S2 NUL presence → S3=0.02
- same 518-dim H/M/T scorer
- nested real-network train sizes: 0 / 25 / 50 / 100 / 200 / 300

Pooled held-out learning curve:

| External train n | Top-1 | Top-3 | Top-5 | Composite |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 32.13% | 48.68% | 58.27% | 2.8417 |
| 25 | 50.60% | 75.06% | 82.73% | 4.3525 |
| 50 | 58.27% | 81.29% | 87.05% | 4.8273 |
| 100 | 66.67% | 87.77% | 92.57% | 5.3477 |
| 200 | 73.92% | 89.95% | 94.74% | 5.7033 |
| 300 | 79.19% | 93.78% | 96.17% | 6.0048 |

Marginal Top-1 gains:
- 0→25: +18.47 pp
- 25→50: +7.67 pp
- 50→100: +8.39 pp
- 100→200: +7.26 pp
- 200→300: +5.26 pp

Interpretation:
- The curve is still rising materially at 300 real-network samples.
- There is no evidence of plateau yet.
- The scorer architecture remains viable; more real-network training data is
  still converting directly into held-out accuracy.
- Therefore do not redesign the representation and do not freeze final
  parameter tables yet.
- Next comparison should benchmark this held-out real-network model directly
  against chardet 7 and charset-normalizer on the exact same test folds.


## Held-out real-network baseline benchmark — current competitive position

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37260216318

Setup:
- Common Crawl `CC-MAIN-2026-39`
- 420 high-confidence real HTTP responses
- 43 WARC files
- same WARC-path 4-fold holdout as the learning-curve experiment
- our selector fixed at S1 UTF-8 validity → S2 NUL presence → S3=0.02
- our scorer uses the same 518-dim H/M/T representation
- up to 300 opposite-fold real-network training samples
- baselines: chardet 7.6.0 and charset-normalizer 3.5.2 default packaged behavior

Pooled held-out Top-1:

| Detector | Top-1 |
| --- | ---: |
| chardet 7 | **95.95%** |
| charset-normalizer | **87.14%** |
| ours | **79.43%** |

Gap:
- ours vs chardet 7: **-16.53 pp**
- ours vs charset-normalizer: **-7.72 pp**

Our fold Top-1:
- fold 0: 80.82%
- fold 1: 72.22%
- fold 2: 81.36%
- fold 3: 74.03%

chardet 7 fold Top-1:
- 95.21% / 100.00% / 96.05% / 96.15%

charset-normalizer fold Top-1:
- 84.93% / 94.74% / 88.70% / 85.90%

Our own Top-k remains strong:
- fold Top-3 ranges 92.47%–100%
- fold Top-5 ranges 94.52%–100%

Interpretation:
- The current model is **not yet competitive on Top-1**.
- chardet 7 is the clear accuracy leader on this real-network sample.
- charset-normalizer is also ahead by ~7.7 pp.
- However, our Top-3/Top-5 are already high, which strongly suggests the
  representation often retains the correct encoding but the final ranking /
  calibration between nearby encodings is still weak.
- Therefore the next highest-value work is **error/rank analysis and
  calibration**, not adding selector complexity.
- The learning curve also had not plateaued at 300 samples, so more
  real-network data can still help, but simply adding data is not the only
  remaining lever.
- Rough chardet/charset-normalizer timing from this run is not formal
  performance evidence and must not be used for final latency claims.

Next priority:
1. compare Top-1 errors where ours has ground truth in Top-3/Top-5;
2. identify repeated encoding-family confusions (e.g. UTF-8 vs UTF-8-SIG,
   CP125x/ISO-8859, CJK families);
3. test a **small calibration/reranking layer** on top of the existing scorer
   before changing the 518-dim representation;
4. keep WARC-path held-out evaluation;
5. only if reranking cannot close a material part of the ~7.7 pp gap to
   charset-normalizer should representation redesign become active again.


## Minimal held-out reranker A/B — validated

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37272317265

Setup:
- same Common Crawl held-out folds
- same selector S1 UTF-8 validity → S2 NUL → S3=0.02
- same 518-dim scorer
- up to 300 opposite-fold external training samples
- minimal deterministic / explainable reranker only

Rules tested:
1. UTF-8 BOM → prefer UTF-8-SIG if already ranked
2. strict UTF-8-valid non-BOM bytes → prefer UTF-8 if already ranked
3. ASCII-only → prefer ASCII if already ranked
4. tiny within-top3 ISO-8859-1 calibration against ISO-8859-9 / CP1257

Pooled results:

| Model | Top-1 | Top-3 | Top-5 | Composite |
| --- | ---: | ---: | ---: | ---: |
| base | 79.43% | 93.78% | 96.17% | 6.0144 |
| reranked | **83.73%** | **94.98%** | **96.65%** | **6.2153** |

Delta:
- Top-1: **+4.31 pp**
- Top-3: **+1.20 pp**
- Top-5: **+0.48 pp**
- composite: **+0.2010**

Decision changes:
- changed Top-1: 37
- beneficial: 24
- harmful: 6
- neutral: 7

All 4 folds improved Top-1:
- fold 0: 80.82% → 84.93%
- fold 1: 72.22% → 77.78%
- fold 2: 81.36% → 84.75%
- fold 3: 74.03% → 80.52%

Interpretation:
- A small reranking/calibration layer is clearly justified.
- Ranking/calibration was a real source of error, not just noise.
- The current minimal reranker recovers ~4.3 pp Top-1 without representation redesign.
- It still trails charset-normalizer 87.14% by ~3.41 pp and chardet 95.95% by ~12.22 pp.
- Next work should refine reranking using held-out evidence, especially the remaining Top-2/Top-3 recoverable errors, but avoid broad heuristic growth.
- Keep rules only if they are cross-fold positive and low-harm.

Next priority:
1. decompose remaining post-reranker errors;
2. identify the next 1–2 highest-frequency recoverable confusion pairs;
3. test only those as separate ablations;
4. prefer deterministic evidence / score-margin calibration over adding many hand-written exceptions.


## Hybrid reranker + candidate calibrator — new baseline

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37275226855

Setup:
- same 420-response Common Crawl held-out set
- same 43 WARC files
- same outer 4-fold WARC-path holdout
- same S1/S2/S3=0.02 selector
- same 518-dim scorer
- up to 300 opposite-fold external training samples
- current rule reranker first; candidate calibrator only acts when the rule
  reranker leaves Top-1 unchanged

Pooled Top-1:
- base scorer: 79.43%
- rule reranker: 83.73%
- candidate calibrator alone: 83.01%
- **hybrid: 85.17%**

Hybrid vs rule:
- **+1.44 pp**
- 52 Top-1 changes
- 31 beneficial
- 7 harmful

Outer folds:
- fold 0: rule 84.93% → hybrid **86.30%**
- fold 1: 77.78% → **77.78%**
- fold 2: 84.75% → **85.88%**
- fold 3: 80.52% → **83.12%**

Decision gate passed:
- pooled hybrid > rule
- 4/4 folds non-negative versus rule

Competitive gap:
- charset-normalizer reference Top-1: 87.14%
- hybrid Top-1: 85.17%
- remaining gap: **1.98 pp**
- on 418 evaluated samples, approximately 9 additional correct Top-1 decisions
  would be enough to match/exceed 87.14%.

Interpretation:
- Hybrid is now the preferred reranking baseline.
- The remaining gap is small enough that another representation redesign is
  not justified yet.
- Next work should decompose **post-hybrid** errors and look for the next
  smallest high-confidence correction mechanism.


## Single-corpus triad confidence sweep — stable 86.12% plateau

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37281853436

Corpus fingerprint:
`f97f0bc6e693e6120574c4eae9cf42d20e0df575017d209294246e939c85aec1`

All confidence thresholds were evaluated in one process against the exact same
420-response corpus and the same fitted models.

Best plateau:
- 0.45: **86.12%**
- 0.475: **86.12%**
- 0.50: **86.12%**
- 0.525: **86.12%**

Each of those produced:
- 10 beneficial changes
- 6 harmful changes
- 22 total changes
- 418 evaluated samples

Decision:
- Treat 0.45–0.525 as a stable plateau rather than a single tuned point.
- Use **0.50** as the representative runtime gate because it sits in the middle
  of the plateau.
- New candidate Top-1 baseline: **86.12%**
- charset-normalizer reference: 87.14%
- remaining gap: **1.02 pp**, approximately 5 additional correct decisions on
  this 418-sample evaluation set.

Do not continue fine-grained threshold tuning around 0.50 unless independent
data invalidates the plateau. Next priority is post-gate error decomposition.


## Canonical final composition — 87.08% Top-1

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37289061345

Corpus fingerprint:
`f97f0bc6e693e6120574c4eae9cf42d20e0df575017d209294246e939c85aec1`

Canonical same-process results:
- baseline: 85.89%
- UTF-8 / UTF-8-SIG specialist only: 86.12%
- UTF-8 / GB18030 specialist only: 86.60%
- GB → SIG composition: 86.84%
- **SIG → GB composition: 87.08%**

SIG → GB changes:
- 12 total Top-1 changes
- 7 beneficial
- 2 harmful
- net +5 correct decisions over the canonical baseline

Fold behavior for SIG → GB:
- fold 0: 85.62% → 86.30%
- fold 1: 88.89% → 88.89%
- fold 2: 85.88% → 87.57%
- fold 3: 85.71% → 87.01%

Reference charset-normalizer Top-1:
- 87.14%

Remaining gap:
- about 0.06 pp
- effectively one additional correct decision on 418 evaluated samples

Decision:
- Promote SIG → GB as the current strongest candidate composition.
- Do not add broad complexity.
- Next step: decompose post-composition errors and look for one narrowly
  supported, cross-fold-safe correction.


## Milestone crossed — replacement-rate guard reaches 87.32% Top-1

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37303010051

Canonical result:
- evaluated: 418
- hits: 365
- Top-1: **87.32%**
- charset-normalizer reference: **87.14%**
- margin: **+0.18 pp**
- beneficial corrections: 5
- harmful corrections: 0

Stable threshold plateau:
- 0.001 through 0.05 all produced the same 365/418 Top-1
- 0.10 and 0.20 regressed to 364/418

Interpretation:
- A hard strict-UTF-8 guard was too aggressive.
- A soft UTF-8 replacement-rate guard separates mildly damaged UTF-8 from
  true non-UTF8 cases more effectively.
- The useful region is broad rather than a tuned knife-edge.

Candidate runtime setting:
- use **replacement_rate <= 0.02** to permit the GB18030 -> UTF-8 specialist
  correction.
- 0.02 is inside the stable 0.001–0.05 plateau and avoids boundary tuning.

Decision:
- Milestone achieved on the current canonical held-out benchmark.
- Do not continue adding complexity before stability/reproducibility validation.
- Next step: rerun the full canonical pipeline with the 0.02 guard and confirm
  fold behavior / reproducibility before locking the baseline.


## Runtime audit — repeated scorer work is now the highest-leverage target

Static inspection after the 87.32% accuracy milestone identified a major
composition artifact in the current experimental inference path.

Current logical path roughly does:
1. base rank / score for hybrid
2. triad gate calls raw rank again
3. triad score_map calls scorer again
4. UTF-8 / UTF-8-SIG pair calls raw rank again
5. that pair score_map calls scorer again
6. UTF-8 / GB18030 pair calls raw rank again
7. that pair score_map calls scorer again

So the same sample can execute the base scorer pipeline about **7 times**:
- H/M/T 768-byte sampling
- 256-bin unigram histogram
- 256-bin hashed bigram histogram
- scalar extraction
- StandardScaler transform
- logistic decision_function

This is not algorithmically necessary.

Additional duplicate work:
- strict UTF-8 checks are repeated across rule/calibrator/specialists;
- BOM and ASCII-only checks are repeated;
- high-byte ratio and NUL ratio are rescanned even though corresponding
  statistics are already produced inside `scorer_features()`;
- pair/triad specialists reconstruct score maps from the same base model output.

### P1 — single-evaluation inference context

Build one immutable per-sample context containing at least:
- sampled H/M/T bytes / feature vector;
- raw class scores;
- sorted rank;
- class -> score map;
- UTF-8 BOM flag;
- strict UTF-8 result;
- ASCII-only flag;
- high-byte ratio;
- NUL ratio;
- UTF-8 replacement/error-rate signal;
- route.

Then change all downstream stages to consume the context rather than recompute
features or model scores.

Target invariant:
- **exact same Top-k output as the current canonical pipeline**
- base scorer feature extraction + model decision function: **1x/sample**
- one shared byte-analysis pass where practical

This is a zero/reuse-cost optimization in the Toolkit sense: work already paid
upstream becomes reusable state instead of being recomputed downstream.

Expected value:
- large runtime reduction is plausible because scorer evaluation currently
  dominates repeated Python/Numpy work;
- exact gain is not yet measured and must not be claimed before A/B benchmark;
- accuracy should be bit/output identical because P1 changes dataflow/reuse,
  not model parameters or decision policy.

Validation plan once Actions capacity is available:
1. output-equivalence test, old vs cached pipeline, every held-out sample;
2. assert rank / Top-1 / Top-3 / Top-5 identity;
3. count scorer-feature and decision-function invocations;
4. benchmark end-to-end per-sample runtime;
5. only then consider P2 native/LUT kernel work.

Do not optimize the native scorer kernel before P1 removes redundant scorer
invocations; otherwise benchmark effort would optimize work that should not
exist.


## P1 locked — cached inference context

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37316898233

Canonical accuracy:
- evaluated: 418
- hits: 365
- Top-1: **87.3206%**
- charset-normalizer reference: **87.1429%**
- exact ranking mismatches versus canonical old path: **0**

Scorer work:
- old `scorer_features()` calls: 1463 / 418 = **3.5 per sample**
- cached calls: 418 / 418 = **1.0 per sample**
- feature/scorer invocation reduction: **71.43%**

Runtime, pooled over 7 timing repeats:
- old: **~2.021 ms/sample**
- cached: **~1.113 ms/sample**
- speedup: **1.816x**
- runtime reduction: **~44.9%**

Fold speedups:
- fold 0: 1.828x
- fold 1: 1.677x
- fold 2: 1.902x
- fold 3: 1.637x

Decision:
- **P1 is accepted and locked.**
- The new runtime baseline is the canonical 87.32% pipeline with a single
  per-sample inference context.
- All downstream reranker/specialist stages must reuse the same base feature
  vector, raw scores, rank, score map, and byte-analysis signals.
- Do not reintroduce duplicate base scorer evaluation.

### P2 — scorer kernel optimization

Next objective:
- preserve **365/418 Top-1** and **0 ranking mismatch**
- preserve **1 base scorer evaluation per sample**
- reduce the cost of building the 518-dim H/M/T feature vector and applying the
  linear scorer.

Priority:
1. profile the cached pipeline and isolate scorer-kernel share;
2. optimize H/M/T byte statistics with the smallest coherent implementation;
3. prefer reuse / LUT / integer-count paths before native extensions;
4. validate exact feature/ranking equivalence;
5. benchmark end-to-end, not microbench only.

Do not change model weights, routing, calibration, specialists, or the 0.02
replacement-rate guard during P2.


## P2 locked — full-pipeline fused runtime baseline

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37320147496

Winner:
- `all_combined`

Correctness:
- evaluated: 418
- hits: **365**
- Top-1: **87.3206%**
- final ranking mismatches: **0**
- base rank mismatches: **0**

Runtime:
- P2 baseline: **~1.141 ms/sample**
- all_combined: **~0.231 ms/sample**
- speedup: **4.944x**
- runtime reduction: **79.77%**

Component results:
- byte_fast: 1.054x
- feature_combined: 1.036x
- linear_fused: 1.294x
- downstream_fused: 1.845x
- downstream + linear: 3.722x
- downstream + linear + byte: 4.304x
- all_combined: 4.944x

Decision:
- **P2 is accepted and locked.**
- The new runtime baseline is `all_combined`.
- Preserve the fused scaler/linear path for both base scorer and downstream
  calibrator/specialists.
- Preserve exact 365/418 accuracy and zero ranking mismatch.

### P3 — Python/control-path overhead reduction

Next objective:
- keep **365/418** and **0 mismatch**
- reduce the remaining ~0.231 ms/sample without changing model behavior

Candidate families to test in one multi-hop tournament:
1. single byte-analysis pass reused for UTF-8/BOM/ASCII/high/nul/replacement signals;
2. avoid per-sample dict construction for score_map; use indexed arrays / cached class indices;
3. avoid full argsort when only top-3 plus full stable ranking reconstruction is needed;
4. precompute downstream feature layout/index maps once per fitted fold;
5. reduce temporary NumPy array/object construction in candidate/triad/pair features;
6. combine the best safe candidates and benchmark end-to-end.

Acceptance:
- exact final ranking equivalence on all 418 samples;
- 365 hits;
- choose the fastest pooled candidate.


## P2 locked — full-pipeline fusion winner

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37320147496

Winner: `all_combined`

Correctness:
- evaluated: 418
- hits: 365
- Top-1: **87.3206%**
- final ranking mismatches versus P1 canonical baseline: **0**
- base-rank mismatches: **0**

P2 pooled runtime:
- P1 cached baseline: **~1.141 ms/sample**
- P2 all-combined: **~0.231 ms/sample**
- speedup vs P1 cached baseline: **4.944x**
- runtime reduction vs P1 cached baseline: **79.77%**

Fold speedups were consistently ~4.8–5.0x.

Main contributors:
- downstream estimator fusion: ~1.85x
- base linear scorer fusion: ~1.29x alone
- byte-analysis fast path: ~1.05x alone
- feature-combined path: ~1.04x alone
- downstream + linear: ~3.72x
- downstream + linear + byte: ~4.30x
- all combined: **4.94x**

Decision:
- **P2 is accepted and locked.**
- New runtime baseline = P2 `all_combined`.
- Preserve exact ranking identity and 365/418 accuracy in all later work.
- GitHub Actions in this public repo is the benchmark authority; do not use the user's local computer as a validation environment.

### P3 direction

Use multi-hop tournaments, not one-hypothesis-per-run.

Priority candidates:
1. remove residual Python object/dict/list construction in the hot path;
2. precompute immutable class/family/route indices and specialist feature layouts;
3. replace repeated generic NumPy/sklearn-shaped helper work with fixed-shape arithmetic;
4. reduce repeated UTF-8 / replacement-rate scans via a single byte-analysis pass;
5. explore batch/fixed-width scorer paths only if single-sample semantics remain identical.

Acceptance gate:
- 365/418 hits;
- final ranking mismatch = 0 on all 418 samples;
- no base-rank mismatch;
- measurable pooled end-to-end speedup over P2;
- validate on public GitHub Actions only.


## P3 locked — indexed branch-precomputed direct-scalar inference

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37325109660

Winner: `indexed_branch_precompute`

Correctness:
- evaluated: 418
- hits: 365
- Top-1: **87.3206%**
- final ranking mismatches versus P2 canonical runtime baseline: **0**

Runtime:
- pooled: **~77.26 us/sample**
- vs P2 baseline in the same run: **1.992x faster**
- vs P3 direct-scalar-lazy: **1.015x faster**
- P1 -> P3 total improvement: roughly **14.4x faster**
- fold winner runtimes:
  - fold 0: ~75.45 us/sample
  - fold 1: ~83.55 us/sample
  - fold 2: ~76.83 us/sample
  - fold 3: ~80.20 us/sample

Accepted techniques:
- fused scaler + linear models;
- direct scalar logit evaluation for downstream calibrators/specialists;
- lazy UTF-8 replacement-rate evaluation;
- raw-score label lookup via precomputed indices;
- precomputed route/branch weight positions;
- combined scorer feature fast path.

Decision:
- **P3 is accepted and locked.**
- New runtime baseline = `indexed_branch_precompute`.
- Preserve 365/418 and exact final ranking identity in all subsequent work.
- Public CodecCat GitHub Actions remains the sole benchmark authority; do not use the user's local computer for validation.

### P4 direction — residual Python/control-path reduction

Continue with one multi-hop tournament, not sequential one-off experiments.

Priority candidates:
1. avoid full `np.argsort` when only top-k and stable full ranking behavior needed;
2. precompute class-order metadata and specialist membership checks;
3. avoid repeated list copies/remove/insert in rerank stages using index/permutation operations;
4. consolidate UTF-8/BOM/high-byte/NUL analysis into one byte pass where semantics remain exact;
5. test fixed-shape scalar evaluation paths that avoid temporary NumPy arrays in triad/pair softmax/logit logic;
6. only consider native/Cython/Rust after pure-Python/NumPy residual overhead is quantified.

Acceptance:
- 365/418 hits;
- final ranking mismatch = 0;
- measurable pooled end-to-end speedup over P3;
- validate on public GitHub Actions only.


## P4 locked — manual small-ops fast path

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37327524648

Winner: `manual_smallops`

Correctness:
- evaluated: 418
- hits: 365
- Top-1: **87.3206%**
- final ranking mismatches versus P3 baseline: **0**

Same-run runtime:
- P3 baseline: **~128.68 us/sample**
- P4 manual_smallops: **~117.56 us/sample**
- speedup vs same-run P3 baseline: **1.095x**
- runtime reduction: **~8.64%**

Important interpretation:
- Cross-run absolute microsecond values are not directly comparable because GitHub-hosted
  runner performance varies.
- The accepted P4 evidence is the **same-run relative speedup** with exact ranking identity.

Rejected / separated candidates:
- `no_list_remove`: effectively neutral (~1.0002x)
- `lite_ctx`, `lite_manual`, `all_p4`: 8 ranking mismatches and 370/418 hits.
  These are NOT runtime-baseline candidates, but the repeatable 370/418 result should be
  preserved as a separate accuracy-research lead.

Decision:
- **P4 is accepted and locked.**
- Runtime baseline now includes manual fixed-size small-op selection/softmax logic.
- Preserve 365/418 and exact final ranking identity in performance work.
- Public CodecCat GitHub Actions remains the only benchmark authority.

### P5 direction — residual hot path + harness acceleration

Run one multi-hop tournament, not sequential one-off tests.

Runtime candidates:
1. fixed top-3 extraction / avoid full score-order work where exact final ordering permits;
2. precompute specialist membership flags and route-local indices;
3. collapse remaining tiny NumPy allocations in triad / pair branches;
4. branch-specialize common no-op paths before specialist evaluation;
5. reuse byte-analysis outputs more aggressively without changing semantics.

Research-harness candidates:
6. fit each fold once and reuse fitted state for all candidates in-process;
7. separate training cost from inference timing and report both explicitly;
8. keep all candidate comparisons on identical fitted objects / rows / process.

Acceptance:
- 365/418 hits;
- final ranking mismatch = 0;
- measurable same-run end-to-end speedup over P4;
- public GitHub Actions only.
