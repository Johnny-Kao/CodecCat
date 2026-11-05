# CodecCat nested-OOF candidate ranking gate (2026-11-05)

## Goal

Avoid misleading "99% on my own benchmark" optimization. The target remains
generalization across sources, lengths and source-heldout groups, not memorizing
the 3,092 available fixture rows.

## Experiment

[GitHub Actions #42004202980](https://github.com/Johnny-Kao/CodecCat/actions/runs/42004202980) — PASS.

- Branch `research/encoding-identifiability-audit`.
- Five existing disjoint development panels, fixed seed 20261105.
- Outer chardet source-group folds kept held out from all candidate ranker fitting.
- Within each allowed training portion, two *additional* inner group-heldout L2
  models produced out-of-fold Top-2 rankings and candidate text features.
- Pairwise logistic discriminator uses normalized L2 margin, text-level Unicode
  characteristics, relative language coherence and input length. It does not
  use codec labels or dataset source IDs as inference features.
- Hard gate established *before* evaluation: switch only if model probability
  > 0.80, L2 score gap <=0.50 and original L3 did not already change choice.
- External Enca and charset-normalizer fixtures excluded from training; still
  evaluated by paired full-document text equality.

## Results

| Bucket | Baseline | OOF switcher | Paired fixes / breaks |
|---|---:|---:|---:|
| All | 2914/3092 (94.243%) | 2914/3092 | 0 / 0 |
| Short <=256 | 140/188 | 140/188 | 0 / 0 |
| Long >256 | 2774/2904 | 2774/2904 | 0 / 0 |
| Enca | 45/81 | 45/81 | 0 / 0 |
| Chardet | 2862/3003 | 2862/3003 | 0 / 0 |

Inner OOF discordant prediction counts:
- Outer 0: 928 examples, 82 need switching
- Outer 1: 1080 examples, 90 need switching
- Outer 2: 923 examples, 70 need switching
- Full-chardet external model: 1452 examples, 127 need switching

The evaluation observed **zero changed predictions**. Therefore this candidate
adds cost without measurable accuracy gains and should not be promoted.

## Methodological warning

The fixed 3,092 fixture records and source families have already influenced
multiple architecture choices. They are a development regression suite, not
a new blind confirmation cohort. A different source distribution can make a
94%/95% claim substantially weaker.

The training examples are imbalanced: most situations where top-two candidates
have differing decoded text do *not* require switching. Lowering the 0.80
threshold based on this evaluation would be tuning on the evaluation set, not
credible evidence of generalization.

## Decision

- Keep current research model at 2914/3092; no promotion to Alpha/main.
- Do not change thresholds based on observed five-panel test results.
- Next genuine architecture advancement should be measured on a separate,
  provenance-checked and never-used confirmation corpus with a predeclared
  acceptance criterion, or begin with training-only calibration diagnostics
  before choosing any threshold.
