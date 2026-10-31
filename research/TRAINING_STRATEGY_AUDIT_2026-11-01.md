# CodecCat stable23 — Training-strategy accuracy attribution

Date: 2026-11-01 (JST)  
Status: **Completed development experiment; no model promotion**  
GitHub Actions: https://github.com/Johnny-Kao/CodecCat/actions/runs/41230402172 (**SUCCESS**)  
Dataset fingerprint: `041c363781d9780e7db0e6e88cd0042e0ddddc769f46637214d1974b05a64ee9`

## Research question

Holding the **published Alpha stable23 L2 implementation and its features constant**, can changing the training-data recipe improve **strict decoded-text correctness** on unseen source groups? Do the gains reflect broad generalization or just displacement of mistakes between confusing encodings?

This is **not** an external independent acceptance study: the samples come from the competitor-owned `chardet/test-data` development/training source. None of the validation samples is included in that fold's training dataset by normalized filename-stem source group; however, these groups are imperfect proxies for independent original documents, and near-duplicates were not eliminated by normalized text.

## Immutable configuration

- Model: `src/codeccat/vnext_strong_l2.py` exactly synchronized to `main` blob `33a093371ddf75721ac1e15ae0372afdd4f26525`, with the released UTF-8/UTF-16/UTF-32 BOM behavior.
- Corpus: `chardet/test-data` at `b0c0d206913ba118ffea2f7d306883e7a4a76db7`.
- 3 existing source-group-hash folds; the same 3,086 held-out byte sequences for all strategies; 1,124 distinct group identifiers.
- Same `TrainableDetector.fit(feature_profile="stable23")`, feature extraction, model scoring and codec compatibility checks; no L3 or threshold tuning.
- **Full**: all eligible training rows.
- **Residual**: original training recipe discarding up to 35% of per-class L1-solved examples, with at least two anchors preserved.
- **Balanced**: deterministic per-class cap of 48, source-group round-robin for diversity; this changes both class distribution **and sample count** (important confound).
- Same strict incremental text comparison at the original 4,096-byte-prefix boundary, stripping recognized leading BOM from both truth and inferred decoded strings.
- GitHub Actions Ubuntu Python 3.12, pinned NumPy 2.5.3 / SciPy 1.18.1 / scikit-learn 1.9.1 / joblib 1.6.0. No tests on owner's Mac.

## Results

| Training policy | Correct decoded text | Same-text rate | Exact encoding label |
|---|---:|---:|---:|
| Full | 2,809 / 3,086 | **91.024%** | 2,610 / 3,086 (84.58%) |
| Residual | **2,828 / 3,086** | **91.640%** | 2,620 / 3,086 (84.90%) |
| Balanced, cap 48 | 2,789 / 3,086 | **90.376%** | 2,469 / 3,086 (80.01%) |

Residual vs Full: **56 helps / 37 harms, net +19 correct texts**, +0.616 percentage points. Difference by fold: +15 of 1,138, +4 of 800, +0 of 1,148. Source-group bootstrap, 1,000 resamples: **+0.031 to +1.166 percentage points** (descriptive interval, dependent on imperfect group identity and non-independent corpus).

Balanced vs Full: **61 helps / 81 harms**, net -20 correct texts, -0.648 percentage points. Exact encoding-label accuracy fell more sharply; this is not interchangeable with decoded-text correctness. Group-bootstrap difference in same-text rate: -1.39 to +0.072 percentage points.

Training sizes per heldout fold:
- Fold 0: full 1,948; residual 1,767; balanced 1,399.
- Fold 1: full 2,286; residual 2,077; balanced 1,579.
- Fold 2: full 1,938; residual 1,754; balanced 1,372.

## Error attribution

**The residual improvement does not consistently fix a confusion pair.** For the EBCDIC classes:
- `cp037`, 34 samples: Full **12** correct texts; Residual **30**; Balanced **31**. Original Full mainly selected `cp500` incorrectly (22 cases).
- `cp500`, 25 samples: Full **21** correct texts; Residual **9**; Balanced **5**. Residual incorrectly selected `cp037` for **13** cases, versus **4** in Full.

This is a **decision-boundary tradeoff**, not evidence that EBCDIC is now understood. More fixture-specific rules or upweighting `cp037` would likely just reverse the error.

**Balancing reduced vital Unicode training evidence.** Example `utf-8` (259 held-out samples): Full **247** correct decoded texts, Residual **249**, Balanced **229**; exact labels under Balanced fell to 105 from Full's 242. Many label mismatches still produce identical text, but the 18 additional wrong texts are real. The cap removes 115–138 UTF-8 training examples per fold. This study **does not** identify whether the cause is class-prior changes, total sample loss, or source selection; it tests their combined effect only.

**Persistent Latin-family confusions remain**, unaffected by residual trimming:
- `iso8859-16` (22 samples): **10** correct texts under all three strategies; confused with `iso8859-2` and `cp1250`.
- `cp1250` (44): Full 32, Residual 33, Balanced 32 correct texts, often confused with `iso8859-2`.
- `cp1252` (42): **27** correct decoded texts under all three, often `iso8859-1` / `iso8859-15` / `hp-roman8`.

**False certainty:** All three policies returned an encoding or decodable selection for every row (0 abstentions) despite wrong-text results. `ambiguous` and uncertainty calibration must be tested before treating this as safe automatic conversion.

## Interpretation and next decision

1. **Keep the Alpha's residual default unchanged.** Do not promote the label-capped Balanced recipe.
2. **Do not claim architectural saturation:** the in-source strategy swap gives a small net gain but exposes countervailing errors; one competitor-origin dataset cannot separate missing information from model-capacity limitations.
3. **Do not retune on these folds:** they have been examined repeatedly and are now development diagnostics. Source-group bootstrap cannot fix source-domain bias or near-duplicate leakage.
4. Next accuracy research should emphasize **independently sourced, document-grouped, legally usable** training/verification data. Prioritize contrasting `cp037` vs `cp500` and `iso8859-16` / `iso8859-2` / `cp1250` without changing class heuristics; lock an untouched confirmation source before selecting a new model.
5. Evaluate retraining as a reversible, versioned **model promotion gate** using full vs previous model paired correctness, family-specific regressions, group-level uncertainty and an explicit abstention/risk-coverage contract.
6. **No CPU, speed or RAM optimization** in this phase. Keep `main`, `v0.1.0a1` and PyPI untouched.

## Reproduction and artifacts

- Harness: [`experiments/logic_first/vnext_training_strategy_audit.py`](../experiments/logic_first/vnext_training_strategy_audit.py)
- CI workflow: [`.github/workflows/vnext-training-strategy-audit.yml`](../.github/workflows/vnext-training-strategy-audit.yml)
- Successful [run #41230402172](https://github.com/Johnny-Kao/CodecCat/actions/runs/41230402172) uploads `report.json` and full `paired.jsonl` in `stable23-training-strategy-development-audit`. The first research run (#41230277019) was intentionally canceled after finding its staged BOM runtime did not match the published Alpha; it is **not evidence**.
