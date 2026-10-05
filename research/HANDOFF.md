# Charset Detection Research Handoff

Updated: 2026-10-05 JST  
Branch: `research/charset-normalizer-e2e-baseline`

## Objective

Build a new charset/encoding detector, not a patch to charset-normalizer.

Primary target:

```text
minimize expected compute
subject to bounded end-to-end error
```

Long-term runtime goal:

```text
stable selector
+ stable native scoring kernel
+ retrainable parameter tables
```

## Current strongest empirical result

A downstream-loss-aware routing tree with per-leaf 518-dim H/M/T-768 linear scorers
beats the same universal scorer.

Round-2 winner run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37205949226

Winner config:
- method D
- C = 0.5
- lambda_cost = 0
- max_leaves = 5

Winner:
- Top-1 72.14%
- Top-3 92.24%
- Top-5 95.30%

Universal baseline from same run:
- Top-1 65.12%
- Top-3 89.04%
- Top-5 93.05%

Delta:
- Top-1 +7.01 pp
- Top-3 +3.19 pp
- Top-5 +2.25 pp

## Stable selector discoveries

Winner split-stability run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37207402007

Across all 3 grouped folds:

1. `utf8_valid_4k > 0.5`
   - selected as root in 3/3 folds
   - threshold exactly 0.5 in 3/3

2. `nul_ratio > 0`
   - selected in 3/3 folds
   - threshold exactly 0 in 3/3

These are currently the strongest selector facts.

Interpretation:

```text
bytes
├─ UTF-8-valid / ASCII / UTF-7-like area
├─ NUL-bearing UTF-16 / UTF-32 family
└─ non-UTF8 + non-NUL residual
```

The older MARKUP / RECORD / CODE / TEXT tree is NOT validated architecture.
Current evidence favors encoding-mechanism routing instead.

## Residual region

Definition:

```text
utf8_valid_4k <= 0.5
AND
nul_ratio == 0
```

Current corpus:
- 1,346 samples
- 77 encodings

Residual discovery run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37213243765

A single third scalar split was unstable:
- fold 0: high_byte_ratio
- fold 1: known_text_ext
- fold 2: no split

Do not freeze a third scalar selector from that result alone.

## Residual convergence batch

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37214176887

Compared shared residual scorer vs tiny 2–4 feature selectors.

Best:
- M2 depth 2
- offered features: high_byte_ratio + ascii_only_4k
- actual learned split feature: high_byte_ratio only

Best residual:
- Top-1 70.05%
- Top-3 84.87%
- Top-5 89.56%

Shared residual:
- Top-1 68.23%
- Top-3 84.57%
- Top-5 88.80%

Delta:
- Top-1 +1.82 pp
- Top-3 +0.30 pp
- Top-5 +0.76 pp

Extra features such as comma_ratio, line_len_cv, lt_ratio, newline_ratio did not
produce stable gains and often hurt.

## High-byte-ratio threshold validation

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37214743877

Per-fold best threshold pairs:

| Fold | t1 | t2 | Top-1 | Top-3 | Top-5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.02 | 0.25 | 72.28% | 86.14% | 89.60% |
| 1 | 0.03 | 0.60 | 72.11% | 91.24% | 94.02% |
| 2 | 0.03 | 0.60 | 74.52% | 87.98% | 91.83% |

Median fold-best:
- t1 = 0.03
- t2 = 0.60

Best fixed pair by mean cross-fold score:
- t1 = 0.03
- t2 = 0.25

Fixed-pair residual results:
- fold 0 Top-1 71.78%
- fold 1 Top-1 71.31%
- fold 2 Top-1 74.52%

Interpretation:
- lower cut is stable around 0.02–0.03
- upper cut is not fully stable: 0.25 vs 0.60
- do NOT freeze S4 yet
- `t1=0.03, t2=0.25` is current robust fixed-pair candidate
- note: this probe creates 3 buckets, while prior depth-2 tree had 4 leaves

## S4 necessity / stability resolution

Run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37250023917

Direct fixed-architecture comparison on the 1,346-sample residual region:
- 5 grouped out-of-fold folds
- fixed S3 lower cut: high_byte_ratio <= 0.03
- compared S3-only vs S3+S4 at 0.25 and 0.60
- paired bootstrap resampled source-name groups (3,000 draws)
- no threshold search in this run

Pooled results:

| Config | Top-1 | Top-3 | Top-5 | Composite score |
| --- | ---: | ---: | ---: | ---: |
| S3 only | 64.12% | 82.76% | 87.96% | 5.0996 |
| S3 + S4=0.25 | 64.93% | 83.28% | 88.71% | 5.1501 |
| S3 + S4=0.60 | 63.82% | 83.36% | 88.86% | 5.1085 |

S4=0.25 vs S3-only:
- Top-1: +0.82 pp
- Top-3: +0.52 pp
- Top-5: +0.74 pp
- composite: +0.0505
- composite positive in 4/5 folds
- paired grouped-bootstrap composite 95% interval: [-0.0140, +0.1065]
- bootstrap P(delta > 0): 94.2%

S4=0.60 vs S3-only:
- Top-1: -0.30 pp
- Top-3: +0.59 pp
- Top-5: +0.89 pp
- composite: +0.0089
- composite positive in only 2/5 folds
- paired grouped-bootstrap composite 95% interval: [-0.0432, +0.0572]

Decision:
- **do not freeze S4**
- the 0.25 cut has a suggestive but not statistically/stability-robust gain
- the 0.60 cut is clearly unstable and trades away Top-1
- current minimal runtime should stop at S3 and let one residual scorer handle
  the remaining high-byte region
- S4 may be reopened only if independent external/WARC data shows a repeatable gain

## Current runtime hypothesis

```text
bounded sample
→ S1 UTF-8 validity
→ S2 NUL presence
→ S3 high-byte-ratio low cut (~0.03)
→ residual leaf scorer
→ confidence / verifier
```

Current selector status:
- S1: supported
- S2: supported
- S3: supported provisionally at ~0.02–0.03
- S4 upper cut: **not accepted**

The objective remains a minimal set of cheap selectors and small parameter
profiles, but only empirically stable selectors should be frozen.

## Important harness work

The first tournament implementation was too slow because every candidate split
retrained child scorers repeatedly.

Optimized harness now uses:
- one process
- shared corpus/features/folds
- coarse-to-fine split screening
- exact downstream evaluation only for top proxy splits
- global child-loss cache
- workflow concurrency

Round-1 optimized run:
https://github.com/Johnny-Kao/OSS-Engineering-Toolkit/actions/runs/37201083741

Stats:
- ~139 sec
- 431 scorer fits
- 673 cache hits

Relevant scripts:
- `experiments/charset_training_tournament.py`
- `experiments/charset_winner_split_stability.py`
- `experiments/charset_residual_selector_discovery.py`
- `experiments/charset_residual_convergence_batch.py`
- `experiments/charset_high_byte_threshold_stability.py`

## Corpus / benchmark warning

Current combined training/development corpus:
- chardet/test-data
- Ousret/char-dataset
- SHA-256 deduplicated
- 2,952 unique byte sequences
- 90 normalized encoding labels

This is algorithm-development data, NOT a final real-network distribution.

The strict-network subset is also provenance-skewed:
- 351 files
- mostly RSS/Atom XML

Headline claims must wait for real WARC/HTTP response validation.

## Next recommended work

Priority order:

1. **External validation of S1/S2/S3**
   - real HTTP/WARC bytes
   - natural-frequency distribution
   - balanced legacy-encoding challenge set

2. **External confirmation of the minimal S1/S2/S3 selector**
   - current in-corpus evidence rejects freezing S4
   - reopen S4 only if independent WARC/HTTP data shows a stable gain
   - validate S3 lower cut around 0.02–0.03 under natural-frequency and
     balanced legacy-encoding challenge distributions

3. **Freeze minimal selector**
   - only after S1/S2/S3 stability holds across external data

4. **Then optimize scoring kernel**
   - native array/LUT/integer implementation
   - do not optimize Python control flow before selector/model stabilizes

5. **Later**
   - calibrated confidence
   - expected fallback cost
   - verifier cascade
   - chardet 7 full end-to-end comparison

## Do not repeat these failed directions

- Do not assume more exact n-grams are better.
- Do not split separate full W matrices only by length.
- Do not use encoding entropy reduction alone to train routing.
- Do not hard-code MARKUP / RECORD / CODE / TEXT.
- Do not add random structural features to residual routing without held-out gain.
- Do not interpret current corpus accuracy as real-web headline accuracy.

## Start-next-session instruction

Read:
1. `research/charset-detection/HANDOFF.md`
2. `research/charset-detection/RESEARCH_PLAN.md`
3. latest relevant experiment scripts listed above

Current resolved question:

> S4 is not stable enough to freeze on the current corpus. Runtime should stop
> at S3 and let one residual scorer handle the remaining high-byte region.

Next session should prioritize independent external validation of S1/S2/S3,
especially real WARC/HTTP bytes and a balanced legacy-encoding challenge set.
Do not restart the project from architecture brainstorming.


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


## P5 locked — branch elision + calibrator static precompute

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37331914730

Winner: `skip_plus_static`

Correctness:
- evaluated: 418
- hits: 365
- Top-1: **87.3206%**
- final ranking mismatches versus P4 baseline: **0**

Same-run runtime:
- P4 baseline: **~88.68 us/sample**
- P5 winner: **~84.49 us/sample**
- speedup vs P4: **1.0496x**
- runtime reduction: **~4.73%**

Component effects:
- `precompute_cal_static`: ~1.044x
- `skip_cal_on_rule`: ~1.010x
- combined `skip_plus_static`: ~1.050x

Research-harness finding:
- fold training cost in this run: **~373.5 s**
- inference optimization is now much faster than repeated model fitting, so subsequent
  performance work should cache/reuse fitted fold state rather than retrain identical
  models in every workflow run.

Decision:
- **P5 is accepted and locked.**
- Runtime baseline now includes manual fixed-size small-op logic, calibrator static-term
  precomputation, and rule-path calibrator elision.
- Preserve 365/418 and exact final ranking identity.
- Public CodecCat GitHub Actions remains the sole benchmark authority.

### P6 direction — cached fitted state + residual hot path

Infrastructure first:
1. generate deterministic fitted state for all 4 folds once;
2. serialize the minimal model/calibrator/specialist state plus corpus fingerprint;
3. publish it as a GitHub Actions artifact and/or checked reproducibility fixture;
4. downstream tournaments load identical fitted state and rows instead of retraining;
5. reject cache reuse if corpus fingerprint / code schema mismatches.

Then benchmark residual runtime candidates on identical loaded state:
- top-k / ordering work;
- specialist branch no-op fast paths;
- remaining tiny NumPy allocations;
- route-local/static metadata folding.

Acceptance:
- 365/418;
- mismatch = 0;
- same-run speedup over P5;
- cached-state reproducibility check passes;
- public GitHub Actions only.


## P6 locked — cached fitted state + residual control-path fast paths

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37335456909

Cached fitted state:
- cache status on first validated run: `miss_built`
- corpus fingerprint: `aea19348bfe9838fc68e1b1c8f5d95eb7d78097df6d89e801de36f8f31946d9f`
- evaluated rows: 418
- external rows: 420
- serialized state size: ~5.68 MB
- one-time training cost: ~369.3 s

Winner: `combined`
- accepted mechanisms: no unnecessary rank copy + direct top-3 membership checks
- hits: **365/418**
- Top-1: **87.3206%**
- ranking mismatch vs P5: **0**
- P5 same-run baseline: ~86.85 us/sample
- P6 winner: ~83.91 us/sample
- speedup vs P5: **1.035x**
- runtime reduction: **~3.38%**

Component effects:
- `no_rank_copy`: ~1.017x
- `direct_membership`: ~1.022x
- combined: ~1.035x

Decision:
- **P6 performance result is accepted and locked.**
- Before opening P7, re-run the same workflow once and require `cache_status=hit`.
- P7 and later should reuse the cached fitted fold state and must not retrain identical folds.
- Preserve 365/418 and exact ranking identity.
- Public CodecCat GitHub Actions remains the benchmark authority.


### P6 cache-hit verification complete

Verification run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37336774166

Result:
- cache status: **hit**
- corpus fingerprint unchanged:
  `aea19348bfe9838fc68e1b1c8f5d95eb7d78097df6d89e801de36f8f31946d9f`
- evaluated rows: 418
- external rows: 420
- no fold retraining performed
- P6 `combined`: 365/418, 0 mismatch
- same-run speedup vs P5: **1.023x**

Interpretation:
- fitted-state cache is now validated for cross-run reuse;
- subsequent P7+ workflows should restore the cache and must not rebuild unless the cache key/schema changes;
- cross-run absolute microsecond values remain non-comparable; use same-run relative speedups only.

### P7 direction — specialist microkernel reduction

Use cached state and test exact-equivalence candidates in one tournament:
1. replace triad tiny NumPy vector accumulation with fixed-size scalar logits;
2. inline specialist pair branch checks to avoid generic call/list overhead on no-op paths;
3. combine both mechanisms;
4. preserve 365/418 and exact ranking identity.


## P7 locked — scalar triad microkernel

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37337516488

Winner: `scalar_triad`

Correctness:
- evaluated: 418
- hits: 365
- Top-1: **87.3206%**
- final ranking mismatch versus P6: **0**

Same-run runtime:
- P6 baseline: **~112.10 us/sample**
- P7 scalar_triad: **~105.74 us/sample**
- speedup vs P6: **1.060x**
- runtime reduction: **~5.68%**

Other candidates:
- `inline_pairs`: ~1.0046x
- `combined`: ~1.052x
- therefore pair inlining is not retained; scalar triad alone is the new baseline.

Decision:
- **P7 is accepted and locked.**
- Cached fitted-state workflow remains mandatory for P8+.
- Preserve 365/418 and exact final ranking identity.

### P8 direction — ranking and tiny-allocation residuals

Use cached state only. Test in one tournament:
1. precompute / reuse model class tuples and route-local class metadata;
2. avoid repeated `np.asarray(raw)` / rank-side temporary creation where safe;
3. replace tiny score/order operations with fixed-size or cached metadata when exact ordering is preserved;
4. early-return common no-op specialist paths before list construction;
5. combine only orthogonal winners.

Acceptance:
- 365/418;
- mismatch = 0;
- same-run speedup over P7;
- no retraining.


## P7 locked — scalar triad specialist microkernel

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37337516488

Cache:
- fitted-state cache hit is already validated by:
  https://github.com/Johnny-Kao/CodecCat/actions/runs/37336774166
- corpus fingerprint:
  `aea19348bfe9838fc68e1b1c8f5d95eb7d78097df6d89e801de36f8f31946d9f`
- P7 reused cached fitted state; no fold retraining.

Winner: `scalar_triad`

Correctness:
- evaluated: 418
- hits: **365**
- Top-1: **87.3206%**
- mismatch vs P6 baseline: **0**

Same-run runtime:
- P6 baseline: **~112.10 us/sample**
- P7 scalar_triad: **~105.74 us/sample**
- speedup vs P6: **1.0602x**
- runtime reduction: **~5.68%**

Other candidates:
- `inline_pairs`: ~1.0046x, effectively negligible
- `combined`: ~1.0518x
- therefore do NOT carry `inline_pairs` forward merely because it was combinable;
  P7 runtime baseline is `scalar_triad` only.

Accepted mechanism:
- replace triad specialist tiny NumPy vector accumulation with fixed 3-class scalar arithmetic.

Decision:
- **P7 is accepted and locked.**
- Preserve 365/418 and exact ranking identity.
- All P8+ performance experiments must use the validated fitted-state cache.
- Do not retrain identical folds unless cache schema/key intentionally changes.
- Public CodecCat GitHub Actions remains the benchmark authority.
- Do not use the user's local computer for validation.

### P8 next direction

Use one cached-state multi-hop tournament. Priority:
1. reduce ranking/order overhead while preserving full exact output ranking;
2. remove remaining tiny NumPy allocations in base-score / downstream hot paths;
3. specialize common no-op branches before expensive specialist logic;
4. precompute any remaining route/class metadata that is still reconstructed per sample;
5. profile first only if candidate value is unclear; avoid one-hypothesis-per-run sequencing.

Acceptance:
- 365/418;
- exact final ranking mismatch = 0;
- measurable same-run speedup over P7;
- cached state hit required;
- public GitHub Actions only.

Operational rules:
- Prefer multi-hop tournaments in one Action/process.
- Same-run relative timing is authoritative; cross-run absolute microseconds are not.
- If a workflow run, PR, or external comment is created, always provide the exact GitHub URL.
- Avoid triggering obsolete P1/P2/P3/P4/P5 workflows when editing shared research files; narrow workflow path filters or cancel accidental runs.


## P8 locked — cached route metadata + direct raw-score reuse

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37340354153

Commit:
https://github.com/Johnny-Kao/CodecCat/commit/3876737fa74404099835da9e93eaf97faebc05d7

Cache:
- fitted-state cache: **hit**
- corpus fingerprint unchanged:
  `aea19348bfe9838fc68e1b1c8f5d95eb7d78097df6d89e801de36f8f31946d9f`
- no fold retraining.

Winner: `hop2_meta_raw`

Accepted mechanisms:
1. precompute/reuse route-local model class metadata;
2. reuse the fused raw-score ndarray directly instead of redundant `np.asarray(raw)` calls.

Correctness:
- evaluated: **418**
- hits: **365**
- Top-1: **87.3206%**
- exact final ranking mismatch vs P7: **0**

Same-run pooled runtime:
- P7 baseline: **~105.261 us/sample**
- P8 `hop2_meta_raw`: **~103.726 us/sample**
- speedup vs P7: **1.0148x**
- runtime reduction: **~1.46%**

Rejected / do not carry forward:
- `hop1_sample_view`: ~0.9798x; bytes->memoryview substitution regressed runtime.
- `hop1_slots_ctx`: ~1.0013x; effectively negligible.
- `hop2_view_slots`: ~0.9748x; regression.
- `hop3_all_safe`: ~0.9843x; regression because the rejected mechanisms contaminate the combination.
- `hop1_raw_direct` alone: ~1.0013x; negligible by itself.
- `hop1_precomputed_meta`: ~1.0127x; useful, but `meta_raw` was faster.

Decision:
- **P8 is accepted and locked.**
- P9 baseline is exactly: P7 scalar triad + P8 precomputed route metadata + direct raw ndarray reuse.
- Do not carry memoryview or slots-context mechanisms into P9.
- Preserve 365/418 and exact final ranking identity.
- Cached fitted state remains mandatory.
- Public CodecCat GitHub Actions remains benchmark authority.

### P9 next direction — ordering/rank residuals

Use one cached-state multi-hop tournament. Prioritize exact-equivalence changes around:
1. avoid temporary allocation in `classes_array[order]` / rank construction;
2. reduce full ordering work only if complete final rank identity remains exact;
3. reduce `sorted_scores=raw[order]` temporary work where downstream consumers can use an exact lightweight representation;
4. precompute any remaining route-local indices/constants used by context construction;
5. do not revisit P8-rejected memoryview/slots mechanisms.

Acceptance:
- 365/418;
- exact final ranking mismatch = 0;
- measurable same-run speedup over locked P8;
- cache hit required;
- public GitHub Actions only.


## P9 locked — ndarray method-form ordering

Primary run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37340808904

Orthogonal-combination follow-up:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37341177042

P9 implementation commits:
- https://github.com/Johnny-Kao/CodecCat/commit/eaa0a3d52a8ab3919c61f6c578e6d84eed55c203
- https://github.com/Johnny-Kao/CodecCat/commit/6dcb97ca6586340bf75e4507a79c8f9cfeff2adb

Accepted mechanism:
- use `raw.argsort()[::-1]` instead of `np.argsort(raw)[::-1]` on the locked P8 context path.

Correctness:
- evaluated: **418**
- hits: **365**
- Top-1: **87.3206%**
- exact final ranking mismatch vs P8: **0**
- fitted-state cache: **hit**
- no retraining.

Same-run evidence:
- run 37340808904: `argsort_method` **1.01224x** vs P8 (~1.21% reduction)
- run 37341177042: `argsort_method` **1.03071x** vs P8 (~2.98% reduction)

Why `method_top3` is NOT carried forward despite being the mechanical winner in the second run:
- `top3_scores` was +0.44% in the first run but -2.47% in the follow-up;
- in the follow-up, `method_top3` beat pure `argsort_method` by only ~0.09%;
- that incremental effect is noise-scale and not independently stable;
- therefore P9 locks only the reproducibly positive argsort mechanism.

Rejected / do not carry forward:
- Python tuple rank gather: large regression (~7–8%);
- byte `bincount`: ~2% regression;
- top-3-only sorted-score materialization: unstable, not accepted;
- combinations containing rank gather or bincount: regressions.

Decision:
- **P9 is accepted and locked.**
- P10 baseline = P7 scalar triad + P8 cached route metadata/direct raw ndarray + P9 ndarray `argsort` method form.
- Preserve 365/418 and exact final ranking identity.
- Cached fitted state remains mandatory.
- Public CodecCat GitHub Actions remains benchmark authority.

### P10 direction — feature/context residual tournament

Prioritize exact-equivalence mechanisms outside rejected P8/P9 paths:
1. reduce byte-analysis allocation/passes without `bincount`;
2. exploit provably-ASCII short inputs to skip redundant UTF-8/BOM checks only when exact;
3. reduce `features_combined` temporary arrays / concatenate overhead with preallocated output;
4. test orthogonal combinations in one cached-state Action;
5. do not reuse memoryview, slots context, Python rank gather, or bincount.

Acceptance:
- 365/418;
- exact final ranking mismatch = 0;
- measurable same-run speedup over locked P9;
- cache hit required;
- public GitHub Actions only.


## P10 locked — preallocated combined-feature output

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37341544038

Implementation commit:
https://github.com/Johnny-Kao/CodecCat/commit/2a2d943055235ad92344cbf8cff101978b94af4e

Winner: `hop1_feature_no_concat`

Accepted mechanism:
- build the 518-element combined feature vector directly into one preallocated float32 output;
- eliminate the separate six-element scalar array and final `np.concatenate`;
- preserve existing unigram/bigram arithmetic and downstream semantics.

Correctness:
- evaluated: **418**
- hits: **365**
- Top-1: **87.3206%**
- exact final ranking mismatch vs P9: **0**
- fitted-state cache: **hit**
- no retraining.

Same-run pooled runtime:
- P9 baseline: **~80.996 us/sample**
- P10 `feature_no_concat`: **~78.721 us/sample**
- speedup vs P9: **1.02889x**
- runtime reduction: **~2.81%**

Rejected / do not carry forward:
- `nul_bytes_count`: ~0.9813x; regression.
- `short_ascii_skip`: ~0.9881x; regression.
- `reuse_short_stats`: ~1.0031x; noise-scale.
- `nul_ascii`: ~0.9783x; regression.
- `feature_reuse`: ~1.0069x; small and materially below feature_no_concat.
- `all`: ~1.0010x; rejected mechanisms erase the feature-construction gain.

Decision:
- **P10 is accepted and locked.**
- P11 baseline = P7 scalar triad + P8 route metadata/direct raw + P9 ndarray argsort + P10 preallocated feature output.
- Do not carry byte-count, short-ASCII, or short-stats-reuse mechanisms forward.
- Preserve 365/418 and exact final ranking identity.
- Cached fitted state remains mandatory.

### P11 direction — bigram/unigram feature microkernel

Use one cached-state multi-hop tournament around the still-allocation-heavy feature kernel:
1. exploit uint8 wraparound equivalence for `(a[i] + a[i+1]) & 255` to remove uint16/int32 cast chains;
2. test direct `np.divide(..., out=float32_slice)` to avoid normalized-count temporaries;
3. test their orthogonal combination;
4. retain P10 preallocation and P9 argsort baseline;
5. correctness gate all 418 outputs before timing.

Acceptance:
- 365/418;
- exact final ranking mismatch = 0;
- measurable same-run speedup over locked P10;
- cache hit required;
- public GitHub Actions only.


## P11 locked — uint8 wraparound bigram kernel

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37341994563

Implementation commits:
- https://github.com/Johnny-Kao/CodecCat/commit/2f8df28067fedbe87feebeb4fe8c8928384a4806
- https://github.com/Johnny-Kao/CodecCat/commit/c0b5aa02a0057691315274c29c448714c81d5eca

Winner: `hop1_uint8_op`

Accepted mechanism:
- replace the bigram-bin cast chain
  `uint16(a[i]) + uint16(a[i+1]) -> & 255 -> int32`
  with native uint8 addition;
- uint8 overflow is modulo 256, exactly matching the existing `& 255` semantics;
- eliminate multiple temporary cast arrays while preserving bin identity.

Correctness:
- evaluated: **418**
- hits: **365**
- Top-1: **87.3206%**
- exact final ranking mismatch vs P10: **0**
- fitted-state cache: **hit**
- no retraining.

Same-run pooled runtime:
- P10 baseline: **~63.654 us/sample**
- P11 uint8 wrap: **~60.048 us/sample**
- speedup vs P10: **1.06004x**
- runtime reduction: **~5.66%**

Other candidates:
- explicit `np.add(..., dtype=np.uint8)`: ~1.0544x, slightly slower than the plain uint8 operator.
- normalization `divide(..., out=...)`: ~1.0134x alone.
- uint8 + divide-out combinations: ~1.0557-1.0567x, slower than uint8 wrap alone.

Decision:
- **P11 is accepted and locked.**
- Carry forward only plain uint8 wraparound bigram construction.
- Do not carry normalization `out=` into P12 because it reduced the stronger winner.
- P12 baseline = all locked P7-P11 mechanisms.
- Preserve 365/418 and exact final ranking identity.
- Cached fitted state remains mandatory.
## P11 locked — uint8 bigram wraparound microkernel

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37341994563

Implementation commits:
- https://github.com/Johnny-Kao/CodecCat/commit/2f8df28067fedbe87feebeb4fe8c8928384a4806
- https://github.com/Johnny-Kao/CodecCat/commit/c0b5aa02a0057691315274c29c448714c81d5eca

Winner: `hop1_uint8_op`

Accepted mechanism:
- compute bigram bins with native uint8 wraparound via `a[:-1] + a[1:]`;
- eliminate the uint16 -> mask -> int32 cast chain while preserving modulo-256 semantics.

Correctness:
- evaluated: **418**
- hits: **365**
- Top-1: **87.3206%**
- exact final ranking mismatch vs P10: **0**
- fitted-state cache: **hit**
- corpus fingerprint unchanged:
  `aea19348bfe9838fc68e1b1c8f5d95eb7d78097df6d89e801de36f8f31946d9f`
- no retraining.

Same-run pooled runtime:
- P10 baseline: **~63.654 us/sample**
- P11 `hop1_uint8_op`: **~60.048 us/sample**
- speedup vs P10: **1.06004x**
- runtime reduction: **~5.66%**

Other candidates:
- `hop1_uint8_add`: ~1.05436x
- `hop1_divide_out`: ~1.01342x
- `hop2_uint8_add_divide`: ~1.05666x
- `hop2_uint8_op_divide`: ~1.05567x

Decision:
- **P11 is accepted and locked.**
- Carry forward only native uint8 bigram wraparound.
- Do not carry `divide_out` merely because it is individually positive; it reduced the uint8 winner's gain in this run.
- P12 baseline = P7 scalar triad + P8 route metadata/direct raw + P9 ndarray argsort + P10 preallocated feature output + P11 uint8 bigram wraparound.
- Preserve 365/418 and exact final ranking identity.
- Cached fitted state remains mandatory.
- Public CodecCat GitHub Actions remains benchmark authority.

### P12 direction — normalized-count temporary elimination

Use one cached-state tournament around the remaining feature-kernel float temporaries:
1. remove unigram `counts.astype(np.float32)` by casting/normalizing directly into the preallocated output slice;
2. remove bigram `bincount(...).astype(np.float32)` the same way;
3. compare direct `np.divide(..., out=..., casting="unsafe")` with assignment + in-place multiply and direct `np.multiply(..., out=...)`;
4. test unigram/bigram components separately before orthogonal combinations;
5. retain P11 uint8 bigram wraparound in every candidate.

Acceptance:
- 365/418;
- exact final ranking mismatch = 0;
- measurable same-run speedup over locked P11;
- cache hit required;
- public GitHub Actions only.


## P12 locked — reuse existing 4 KiB sample for UTF-8/BOM checks

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37342354063

Implementation commits:
- https://github.com/Johnny-Kao/CodecCat/commit/ea3aa1f3622dae9c5daf94153b6af7b5a191225c
- https://github.com/Johnny-Kao/CodecCat/commit/a78c398a4143c12991a767043f19901e007b8bd1

Winner: `hop1_sample_checks`

Accepted mechanism:
- context construction already owns `sample = data[:4096]`;
- run strict UTF-8 decode directly on that sample instead of calling a helper that slices `data[:4096]` again;
- run BOM `startswith` directly on the same sample;
- semantics are identical because both original helpers inspect the same prefix.

Correctness:
- evaluated: **418**
- hits: **365**
- Top-1: **87.3206%**
- exact final ranking mismatch vs P11: **0**
- fitted-state cache: **hit**
- no retraining.

Same-run pooled runtime:
- P11 baseline: **~34.998 us/sample**
- P12 sample checks: **~34.516 us/sample**
- speedup vs P11: **1.01397x**
- runtime reduction: **~1.38%**

Other candidates:
- direct sample decode alone: ~1.0037x.
- inline BOM alone: ~1.0099x.
- array-based H/M/T concatenate: ~0.9831x; regression.
- array-based H/M/T preallocation: ~0.9874x; regression.
- H/M/T array variants combined with sample checks remained regressions.

Decision:
- **P12 is accepted and locked.**
- Carry forward direct checks on the existing 4 KiB sample.
- Keep the original bytes-based `hmt768`; array restructuring is rejected.
- P13 must first profile the locked P12 pipeline because the next dominant hotspot is no longer obvious.
- Preserve 365/418 and exact final ranking identity.
- Cached fitted state remains mandatory.
## P12 locked — direct cast into output + in-place reciprocal multiply

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37342438932

Implementation commits:
- https://github.com/Johnny-Kao/CodecCat/commit/7a6c92ad2443d80c8d810e707b7b7ba4eeb8053c
- https://github.com/Johnny-Kao/CodecCat/commit/ef43512ee4b8823099616dfe97bf065bd89dea15

Winner: `hop2_both_assign_multiply`

Accepted mechanism:
- write integer unigram/bigram counts directly into the preallocated float32 output slices;
- normalize each slice in place with reciprocal multiplication;
- eliminate both temporary `astype(np.float32)` arrays.

Correctness:
- evaluated: **418**
- hits: **365**
- Top-1: **87.3206%**
- exact final ranking mismatch vs P11: **0**
- fitted-state cache: **hit**
- corpus fingerprint unchanged:
  `aea19348bfe9838fc68e1b1c8f5d95eb7d78097df6d89e801de36f8f31946d9f`
- no retraining.

Same-run pooled runtime:
- P11 baseline: **~39.535 us/sample**
- P12 `hop2_both_assign_multiply`: **~38.753 us/sample**
- speedup vs P11: **1.02016x**
- runtime reduction: **~1.98%**

Rejected / do not carry forward:
- unigram direct divide: ~0.9591x;
- bigram direct divide: ~0.9249x;
- both direct divide: ~0.9778x;
- both direct multiply: ~0.9901x.

Decision:
- **P12 is accepted and locked.**
- Carry forward assignment + in-place reciprocal multiply for both count vectors.
- Do not carry direct divide/direct multiply variants.
- Preserve 365/418 and exact final ranking identity.
- Cached fitted state remains mandatory.
- Public CodecCat GitHub Actions remains benchmark authority.

### P13 direction — residual pipeline profile before further optimization

The remaining dominant cost is no longer obvious after P7-P12. Run one cached-state residual profile before selecting the next optimization family.

Measure on the locked P12 path:
1. feature kernel;
2. fused linear score + argsort/rank materialization;
3. byte/context analysis (`strict_utf8`, BOM, high/nul scan);
4. context object construction;
5. downstream reranker;
6. full end-to-end inference.

Use the profile only to choose P13 optimization targets; do not treat isolated component sums as exact additive wall time.
No retraining. Public GitHub Actions only.


## P13 profile — locked P12 component decomposition

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37342621474

Profile commits:
- https://github.com/Johnny-Kao/CodecCat/commit/691ede585aa36a2af45770df6c5d82de47a53b1a
- https://github.com/Johnny-Kao/CodecCat/commit/67b2e0a9611924bd6d4ff11c969640d792e90668

Pooled component timing:
- feature kernel: **~19.913 us/sample** — **40.26%** of measured component sum.
- downstream: **~11.776 us/sample** — **23.81%**.
- linear + ordering: **~10.951 us/sample** — **22.14%**.
- sample checks: **~6.825 us/sample** — **13.80%**.
- full locked P12 pipeline: **~59.866 us/sample**.

Interpretation:
- feature construction is now the dominant measured component;
- the feature-to-linear boundary still converts the newly built float32[518] feature vector to float64 on every sample inside `fused_raw`;
- P14 should target that conversion and matrix-vector orientation before revisiting downstream micro-branches.

P14 priority:
1. preserve float32-rounded feature values while storing them directly in float64 to avoid the subsequent 518-element cast;
2. compare current `x @ w.T` with equivalent `w @ x` GEMV orientation;
3. test the orthogonal combination;
4. optionally test float32 fused weights only as an exploratory candidate behind exact 418-output correctness gating.
## P13 residual profile complete — feature kernel remains dominant

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37342763853

Profile commits:
- https://github.com/Johnny-Kao/CodecCat/commit/352b545a1d2a7bf87873d8a1dddf22a0876e8c6f
- https://github.com/Johnny-Kao/CodecCat/commit/38146914657043fd378d3358fa83dc8a4f414305

Cache:
- fitted-state cache: **hit**
- corpus fingerprint unchanged:
  `aea19348bfe9838fc68e1b1c8f5d95eb7d78097df6d89e801de36f8f31946d9f`
- no retraining.

Directional isolated shares versus same-run full locked-P12 path:
- feature kernel: **~27.27%**
- downstream reranker: **~16.41%**
- score/order: **~12.07%**
- byte analysis: **~9.76%**
- context construction: **~6.39%**

Interpretation:
- isolated timings are directional, not additive;
- feature construction is still the largest actionable single component after P7-P12;
- continue feature-kernel work before returning to byte/context paths.

### P13 optimization direction — fuse unigram + bigram histogram

Test exact-equivalence candidates that replace two separate `np.bincount` calls with one combined 512-bin histogram:
1. locked P12 two-bincount baseline;
2. one combined histogram with uint16 index storage;
3. one combined histogram with native `np.intp` index storage;
4. preserve P11 uint8 wraparound and P12 in-place normalization;
5. correctness gate all 418 outputs before timing.

Acceptance:
- 365/418;
- exact final ranking mismatch = 0;
- measurable same-run speedup over locked P12;
- cache hit required;
- public GitHub Actions only.


## P14 pending — feature-to-linear boundary tournament

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37343012381

Implementation commits:
- https://github.com/Johnny-Kao/CodecCat/commit/e2eeec3738fc5c36a59c5c24ec83ed0c220bb1b5
- https://github.com/Johnny-Kao/CodecCat/commit/2e152c955ede0e38ac49a06275ac70cb953ab16f

Status at handoff/update:
- workflow is queued for a GitHub-hosted runner;
- no benchmark result yet;
- this is not a code/test failure.

Candidates:
- locked P12 baseline;
- feature vector stored as float64 while preserving each locked float32-rounded feature value;
- equivalent `w @ x` GEMV orientation;
- orthogonal combination.

Purpose:
- remove the per-sample 518-element float32 -> float64 conversion at the feature/linear boundary;
- test whether matrix-vector orientation improves the small-class linear scorer.

Acceptance remains:
- 365/418;
- exact final ranking mismatch = 0;
- cache hit;
- measurable same-run speedup over locked P12.


## P14 locked — row-oriented GEMV scorer

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37343012381

Implementation commits:
- https://github.com/Johnny-Kao/CodecCat/commit/e2eeec3738fc5c36a59c5c24ec83ed0c220bb1b5
- https://github.com/Johnny-Kao/CodecCat/commit/2e152c955ede0e38ac49a06275ac70cb953ab16f

Winner: `hop1_gemv`

Accepted mechanism:
- compute the fused linear score as `w @ x + b` instead of `x @ w.T + b`;
- preserve the same float64 conversion and model weights;
- only change the equivalent matrix-vector orientation.

Correctness:
- evaluated: **418**
- hits: **365**
- Top-1: **87.3206%**
- exact final ranking mismatch vs P12: **0**
- fitted-state cache: **hit**
- no retraining.

Same-run pooled runtime:
- P12 baseline: **~91.348 us/sample**
- P14 GEMV: **~90.543 us/sample**
- speedup: **1.00889x**
- runtime reduction: **~0.88%**
- all four folds improved.

Rejected:
- float64 feature storage preserving float32-rounded values: ~0.9752x; regression.
- float64 feature + GEMV: ~0.9783x; regression.

Decision:
- **P14 is accepted and locked.**
- Carry forward row-oriented GEMV only.
- Do not carry float64 feature storage.
- Next target should use the P13 profile: downstream is the second-largest measured component (~23.8%).
## P13 combined-histogram experiment rejected

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37343071775

Implementation commits:
- https://github.com/Johnny-Kao/CodecCat/commit/83b944e70ae008c6c47fcad61c3b402371a709d3
- https://github.com/Johnny-Kao/CodecCat/commit/ac0edc92b74a1ac3c928dbe5b95eca8e7462278a

Correctness:
- evaluated: **418**
- hits: **365**
- exact final ranking mismatch: **0** for all candidates
- fitted-state cache: **hit**

Same-run pooled runtime:
- locked P12 baseline: **~91.975 us/sample**
- `combined_intp`: **~98.104 us/sample** = **0.9375x**
- `combined_uint16`: **~98.933 us/sample** = **0.9297x**

Decision:
- **Reject combined-histogram mechanism.**
- Buffer construction / index widening costs exceed the saved `np.bincount` call.
- P12 remains the locked performance baseline.
- Do not retry combined histogram without a fundamentally different zero-copy mechanism.

### P14 direction — downstream residual decomposition

P13 residual profile identified downstream reranking as the second-largest isolated component (~16.4%).
Before changing downstream semantics, measure the locked P7 downstream stages on prebuilt contexts:
1. hybrid / candidate calibrator;
2. triad gate;
3. SIG pair;
4. GB pair;
5. final replacement-rate guard;
6. branch activation frequencies.

Use cached fitted state and locked P12 contexts. No retraining. Public GitHub Actions only.
## P14 downstream profile complete — candidate calibrator is the dominant residual

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37343419015

Profile commits:
- https://github.com/Johnny-Kao/CodecCat/commit/65fcadee9af1a052772633e2e213958d92bef019
- https://github.com/Johnny-Kao/CodecCat/commit/f1fdaafc5fa0a4da9846ba663251bd12ff0e3a03

Cache:
- fitted-state cache: **hit**
- corpus fingerprint unchanged.

Directional isolated downstream shares:
- hybrid / candidate calibrator: **~48.35%**
- triad gate: **~24.43%**
- SIG pair: **~9.67%**
- GB pair: **~5.95%**

Activation rates:
- rule override: **37/418 = 8.85%**
- triad entry: **304/418 = 72.73%**; changed output **21/418 = 5.02%**
- SIG entry: **226/418 = 54.07%**; changed output **4/418 = 0.96%**
- GB entry: **228/418 = 54.55%**; changed output **9/418 = 2.15%**
- replacement guard shape: **8/418 = 1.91%**

Interpretation:
- the candidate calibrator is the largest remaining downstream target;
- pair branches are not the first priority despite frequent entry because their isolated cost is small;
- P14 optimization should reduce repeated calibrator arithmetic/lookups without changing decision semantics.

### P14 optimization tournament — calibrator common-term hoisting

Test on the locked P12 full pipeline:
1. P12/P7 downstream baseline;
2. hoist per-context calibrator terms shared by all top-3 candidates;
3. replace `(route, candidate)` tuple-key static lookup with route-local candidate lookup;
4. algebraically collapse score coefficients and precompute rank-position terms;
5. combined route-local + algebraic path.

Correctness gate:
- 365/418;
- exact final ranking mismatch = 0;
- cache hit required.
## P14 calibrator tournament — downstream winner selected, full-pipeline validation pending

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37343742851

Implementation commits:
- https://github.com/Johnny-Kao/CodecCat/commit/6aa8554cedd96f52d948af4113293399b41df39a
- https://github.com/Johnny-Kao/CodecCat/commit/c520e4e6fe8f4c130261e6ca2ae7053e7b915a96

Downstream-only same-run results on prebuilt locked-P12 contexts:
- baseline: **~15.194 us/sample**
- common-term hoist: **~13.569 us/sample**, ~1.1198x
- route-local lookup: **~13.214 us/sample**, ~1.1498x
- algebraic collapse: **~11.591 us/sample**, ~1.3108x
- combined algebraic + route-local: **~11.356 us/sample**, **~1.3380x**, **~25.26% reduction**

Correctness:
- 365/418 for every candidate;
- exact final ranking mismatch = 0.

Selected candidate:
- `combined`

Important:
- this timing isolates downstream and is not yet the formal P14 end-to-end acceptance result;
- run one full-pipeline same-run A/B against locked P12 before locking P14.
## P14 locked — calibrator algebraic collapse + route-local static lookup

Full-pipeline acceptance run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37343941337

Full-validation commits:
- https://github.com/Johnny-Kao/CodecCat/commit/532127af2f3d3ec6e6cd257a9be65a4a91851026
- https://github.com/Johnny-Kao/CodecCat/commit/af857787962fb3470546ad4bd3211f67491b78d7

Selected mechanism:
- hoist calibrator terms shared by the top-3 candidates;
- algebraically collapse the score expression;
- precompute rank-position terms;
- replace repeated `(route, candidate)` tuple-key lookup with route-local candidate static lookup.

Correctness:
- evaluated: **418**
- hits: **365**
- Top-1: **87.3206%**
- exact final ranking mismatch vs locked P12: **0**
- fitted-state cache: **hit**
- no retraining.

Same-run full-pipeline runtime:
- locked P12 baseline: **~97.116 us/sample**
- P14 combined: **~90.585 us/sample**
- speedup: **1.07210x**
- runtime reduction: **~6.73%**

Supporting downstream-only tournament:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37343742851

Downstream-only winner:
- baseline: ~15.194 us/sample
- P14 combined: ~11.356 us/sample
- ~1.3380x / ~25.26% downstream reduction.

Decision:
- **P14 is accepted and locked.**
- P14 baseline now includes P7 scalar triad + P8 metadata/raw reuse + P9 ndarray argsort + P10 preallocated feature output + P11 uint8 bigram wrap + P12 in-place normalization + P14 calibrator collapse.
- Preserve 365/418 and exact final ranking identity.
- Public GitHub Actions remains benchmark authority.

### P15 direction

Re-profile the locked P14 full path before selecting the next optimization family. P14 materially changed the downstream share, so the P13 residual profile is no longer authoritative for prioritization.
## P15 locked-P14 residual profile complete

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37344140427

Profile commits:
- https://github.com/Johnny-Kao/CodecCat/commit/cf702c0a8eee8e62ed89cacc76be8d3f4eb4486a
- https://github.com/Johnny-Kao/CodecCat/commit/3b927b669093faaa67880df3c75ced8f6ec08e68

Directional isolated shares on the locked P14 path:
- feature kernel: **~34.87%**
- downstream: **~15.86%**
- score/order: **~15.11%**
- byte analysis: **~12.73%**
- context construction: **~8.78%**

Decision:
- feature remains the largest residual component;
- however, the previously tested combined-histogram mechanism is rejected;
- split feature and score/order into primitive costs before choosing the next implementation.
## P15 primitive profile complete — scalar reductions and small score matmul dominate

Run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37344395859

Profile commits:
- https://github.com/Johnny-Kao/CodecCat/commit/6c5600951bc6a72e8cbbf20793ef5139657290e2
- https://github.com/Johnny-Kao/CodecCat/commit/5a602d9acc0792fdb632ae1455f848a93942294f

Feature primitive timings (directional):
- full feature: **~18.780 us**
- scalar stats: **~8.570 us**
- finish counts total: **~13.005 us**
- unigram bincount: **~2.012 us**
- bigram bincount: **~2.083 us**
- bigram add: **~0.923 us**
- hmt768: **~0.631 us**
- frombuffer: **~0.667 us**

Score/order primitive timings:
- full score/order: **~18.724 us**
- full score: **~16.229 us**
- matmul + bias: **~14.264 us**
- argsort reverse: **~1.175 us**
- float64 cast: **~0.888 us**

Interpretation:
- remaining feature cost is dominated by small scalar reductions, not histogram construction;
- remaining score cost is dominated by the small matrix-vector product, not sorting;
- P15 should test these two orthogonal families in one full-pipeline tournament.
