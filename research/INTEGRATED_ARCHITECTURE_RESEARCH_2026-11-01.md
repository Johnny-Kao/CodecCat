# Integrated CodecCat architecture research — 2026-11-01

Status: **GitHub Actions SUCCESS**, development-only evidence.
Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41308393783
Repro: `experiments/logic_first/integrated_architecture_gate.py`, `.github/workflows/integrated-architecture-gate.yml`.

## Single-run scope

- Pinned `chardet/test-data@b0c0d206913ba118ffea2f7d306883e7a4a76db7` grouped 3-fold development evaluation, original stable23 residual recipe. No hyperparameter change.
- Strict decoded-text classification of all held-out mistakes, mutually exclusive by rule.
- Portable save/reload and exact decision equivalence on **every held-out sample**, independently for each fold.
- Third-party uchardet fixture comparison with pinned `chardet==7.0.1` and `charset-normalizer==3.4.4`, text-equivalence de-dup against training-origin corpus.

## Results

| Item | Observed |
|---|---:|
| Grouped development samples | 3,086 |
| Correct decoded text | 2,830 (91.70%) |
| Wrong decoded text | 256 |
| Wrong: truth in validated top-three, but not selected | 141 |
| Wrong: truth absent from validated top-three (but L1 theoretically possible) | 111 |
| Wrong: declared truth structurally excluded | 4 |
| Portable model roundtrip identical decision/candidate/ambiguity | **3,086/3,086** |
| Per-fold training seconds | 6.181 / 6.972 / 6.322 |
| Per-fold portable bytes | 10,943,962 / 11,822,802 / 11,366,116 |
| External semantic non-overlap fixture samples | 21 |
| External correct Unicode decoding | CodecCat 20, chardet 19, charset-normalizer 21 |

The structural exclusion count of four concerns **wrong-text cases only**, unlike the prior L1-only audit reporting seven truth-label exclusions across all 3,086 samples. These are different denominators; do not confuse them.

No actual architecture B/C implementation was compared here. This **establishes the baseline, error distribution, and portable model contract only**. L3/rejection, CPU/peak RSS, source-authenticated blind evaluation, retraining on unfamiliar data and rollback remain unverified. This report must not claim those tasks completed.

## Decision

1. **Do not tune classifier parameters or promote a new model** based on benchmark fixtures.
2. The dominant development errors require examination of (a) cases where the correct codec is already in validated top-three (141), and (b) cases lost in ranking (111). Do not interpret all 252 as fixable: encoding aliases and ambiguous actual texts require case-level evidence.
3. Examine four wrong-text structural exclusions for label normalization/truncated-prefix/codec decoder issues before changing L1. No blanket hardcoded family rules.
4. The three-fold save-load parity supports keeping the current portable model format. Further lifecycle needs adversarial samples, rejection checks and versioned rollback.
5. Independent natural-document provenance and non-overlap remain critical before saying performance is on par with chardet 7.

## Constraints

Only research branch `research/encoding-identifiability-audit` was changed. No modification to main, published alpha, PyPI, or upstream. All execution in GitHub Actions.
