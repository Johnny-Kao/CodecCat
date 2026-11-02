# Architecture-first accuracy attribution — 2026-11-02

## Sources and verification
- GitHub Actions: https://github.com/Johnny-Kao/CodecCat/actions/runs/41385752502 — PASS.
- 3,086 held-out grouped-fold cases, fixed model within each fold. Compare `rank(data)` vs `rank(data, structural_first=True)`, identical parameters.
- chardet 7's 13-stage architecture is a sequence of detection and rejection operations, not a 13-layer learned model. See https://pypi.org/project/chardet/7.5.0/ and https://docs.rs/crate/chardet-rs/latest.

## Measured results
| | Stable23 | Binary TF alpha 0.0003 |
|---|---:|---:|
| held-out | 3086 | 3086 |
| normal correct | 2828 | 2897 |
| structural_first correct | 2828 | 2897 |
| predictions changed by structural_first | 0 | 0 |
| mistakes with text-equivalent answer among returned Top3 | 153 | 149 |
| mistakes with no text-equivalent Top3 | 105 | 40 |
| no-candidate errors | 0 | 0 |

**Implications**
1. The `structural_first` flag only uses the existing strict codec-decoding possible set. Normal ranking already applies strict validation lazily. The complete lack of change is consistent with the two being logically redundant for the tested data. Adding more code to this flag does not improve accuracy.
2. In Binary TF, 149/189 wrong cases could be corrected by a **perfect** Top3 reranker, giving a non-achievable oracle upper bound of 3046/3086 (98.70%); real general-purpose rerankers will recover fewer cases and may damage some currently-correct outcomes. The other 40 failures require candidate generation/features/structural evidence or expanded coverage, not a Top3-only reranker.
3. Binary false-acceptance is separate: previous synthetic negative corpus 98/98 accepted; no calibration or negative abstain stage exists.
4. Original chardet's documented 13 stages are not a prescription to implement thirteen modules. Distinguish BOM/escape/Unicode gates already present in CodecCat from missing *independent evidence* (markup declarations, binary MIME guard, CJK structural admissibility, ambiguity resolution).
5. A combined architecture should have explicit stage contracts: deterministic evidence -> candidate eligibility -> statistical ranking -> secondary independent evidence -> accept/abstain; isolate each stage's accuracy/latency and regressions.

## Next experiment and promotion gates
- Freeze Binary TF weights; add **offline** Top3 secondary-evidence reranker, with independent signals derived from decoded text, document declarations and mixed scripts. Avoid using truth labels at inference. Evaluate a paired oracle-diagnostic and honest candidate policy separately.
- Cross-fit any learned reranker on source-disjoint folds; no training and evaluation on the same held-out predictions.
- Report positive gains, losses, per-codec regressions, external-source validation, and CPU/RAM.
- Keep `main` and released `v0.1.0a2` unchanged.
