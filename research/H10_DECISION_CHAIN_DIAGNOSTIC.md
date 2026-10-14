# H10 Decision Chain Attribution — Final Diagnostic

Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/38583508383
Source: H7 immutable artifact from run 38553419903, n=1,560. All data are research-exposed.
Status: diagnostic completed; no candidate promoted; R21 unchanged.

## Findings
- Raw scorer: 1,405/1,560 (90.06%)
- Baseline final before escalation: 1,411/1,560 (90.45%)
- Frozen end-to-end R21: 1,446/1,560 (92.69%)
- Raw Top-3 truth coverage: 1,502/1,560 (96.28%). This is a **raw Top-3 reranker ceiling**, not a full-architecture ceiling.
- Raw correct then final wrong: 35 samples; raw wrong then final correct: 76 samples. Net end-to-end +41.

Sequential baseline-stack transitions:
| Step | Beneficial | Harmful | Net |
|---|---:|---:|---:|
| raw -> hybrid | 46 | 48 | -2 |
| hybrid -> triad | 10 | 14 | -4 |
| triad -> UTF8/SIG | 12 | 0 | +12 |
| UTF8/SIG -> UTF8/GB | 2 | 3 | -1 |

The Full-B traces were computed for eligible routes even where production escalation would not select them. Transitions through full_raw/full_final are *observational comparisons*, NOT the causal effects of disabling or enabling a single production stage.

By route:
| Route | N | Raw Top-1 | Final Top-1 | Raw Top-3 | Raw correct->final wrong | Raw wrong->final correct |
|---|---:|---:|---:|---:|---:|---:|
| U | 477 | 453 | 459 | 475 | 11 | 17 |
| RH | 917 | 827 | 847 | 886 | 21 | 41 |
| RL | 161 | 123 | 137 | 137 | 3 | 17 |
| N | 5 | 2 | 3 | 4 | 0 | 1 |

**RH x CJK:** n=220, raw 182, final 192, raw Top-3 coverage 201; downstream corrected 10 and harmed **zero** raw-correct CJK rows. This contradicts a blanket hypothesis that downstream is responsible for CJK weakness. Among the final 28 CJK errors, 19 lack truth in raw Top-3, so Top-3-only reranking cannot resolve them; re-ranking can address at most 9 additional errors assuming fixed raw candidate membership.

**Domain-specific downstream conflict:** on CC34 + CC39, raw 520/521, final 505/521 (15 raw-correct made wrong and no corrected wrong predictions). On CC08 + CC12, raw 429/519, final 459/519 (+30).

## Recommended next controlled architectural study
Prioritize new **RH/CJK candidate discovery or discriminative evidence**, rather than blanket suppression of downstream logic. Design feature-level representation with correct alignment to a newly trained *separate candidate* model; immutable R21 remains baseline. Diagnose and isolate raw Top-3 missing cases and test independently of the benchmark-selected crawls. A second small candidate can test delegating override authority conditionally, but must be justified by structural evidence rather than fitted per-crawl gates.

Do not claim a successful improvement, release readiness, or untouched holdout status from this diagnostic.
