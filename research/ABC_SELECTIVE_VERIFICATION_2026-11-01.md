# CodecCat A/B/C selective verification experiment — 2026-11-01

Status: **GitHub Actions SUCCESS, development-only**.
Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41310371634

## Compared

- A: stable23 residual model, unchanged.
- B: same trained model with existing `structural_first=True` control. **Not new CJK/UTF evidence.**
- C: B plus existing `LanguageCoherence.fit(...).rerank(...)`, predeclared default gates, trained exclusively from each fold's training rows.

3,086 grouped held-out development examples, same pinned `chardet/test-data`. No SGD tuning.

| Metric | A | B | C |
|---|---:|---:|---:|
| Correct decoded texts | 2,828 | 2,828 | **2,838** |
| Incorrect decoded texts | 258 | 258 | **248** |
| Abstentions | 0 | 0 | 0 |

L3 verifier invoked on 164/3,086, changed 16 predictions. Ten changed predictions repaired wrong text; **zero previously correct predictions became incorrect text**. The other six changes did not change correct/wrong-text status. This is preliminary evidence only: one development corpus already used in many studies. It does NOT justify automatic production enablement.

## Decision

1. Existing structural-first validation is not worthwhile as accuracy optimization; A=B.
2. Existing selective L3 has a promising development-only +10 net correct texts at sparse invocation frequency, but needs genuinely independent tests, CPU/RAM measurement, and exact source-level provenance before integration.
3. Do not introduce a large rules engine or retune SGD to optimize this fixture corpus.
4. Save/load of L3 and opt-in operation deserve focused product-contract testing. Reject/ambiguity calibration is still missing and has **not** been tested in C.
5. No changes to main, alpha, packaging or released model.

## Reproduction

- `experiments/logic_first/abc_architecture_evidence.py`
- `.github/workflows/abc-architecture-evidence.yml`
- Artifact `abc-architecture-evidence` with aggregated results and per-case changes.
