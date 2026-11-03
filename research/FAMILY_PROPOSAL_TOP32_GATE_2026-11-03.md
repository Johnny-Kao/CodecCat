# Family proposal Top32 architecture gate — 2026-11-03

## Implementation
- Research-only candidate family proposal: `src/codeccat/vnext_family_proposal.py`.
- Paired three-fold development + immutable Enca external audit: `experiments/logic_first/family_proposal_gate.py`.
- GitHub Actions https://github.com/Johnny-Kao/CodecCat/actions/runs/41581240661 — PASS.
- Unmodified Binary TF 1–2 gram / alpha=.0003. Independent LanguageCoherence trained on fold training data; candidates derived from strict-valid learned codec full-score order, up to 32 distinct codec labels. Two policies chosen a priori, not fitted to Enca: conservative language advantage >1.0 and no script-coherence deterioration, script_strong language advantage >.6 and script gain >=.2.

## Paired results
| Source | base | conservative | fixed | broken | changed | p50 extra CPU | p95 extra CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| development 3086 | 2897 | 2898 | 3 | 2 | 5 | 60.079 ms | 160.184 ms |
| Enca 81 | 45 | 45 | 0 | 0 | 3 | 5.241 ms | 9.508 ms |

Script-strong policy: unchanged 2897/3086 and 45/81, no changed predictions; similar CPU overhead.

## Interpretation
- The Top32 expansion **does not** cure Enca's 28 missing-Top3 cases: no net improvement despite scanning candidates.
- Expensive, weak universal Unicode text coherence is inadequate to reliably identify visually valid but semantically wrong decodings across codepages and languages.
- CPU cost ~60 ms median additional work on development corpus is far above acceptable for this product. Baseline ranking cost is excluded from this measurement.
- The `no script deterioration` condition is not a separate script-based proposal generator; it is only a non-degradation filter on the language model's proposal. The module is a research prototype, not evidence that a true script-specific family selector has been tested.
- Many Enca samples stem from shared original texts across different encodings; group-level text-source analysis is a prerequisite to population generalization.
- No production code changes, published `v0.1.0a2` unchanged.

## Decision
**FAIL / do not promote**. No further scalar threshold tuning to improve the Enca holdout. Reconsider cheap byte-specific per-family evidence, or separate family classifier trained on source-disjoint data, with a hard latency budget and separately evaluated candidate coverage. Avoid 32-way full-language-scoring architecture.
