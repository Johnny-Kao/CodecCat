# H6 Outcome — Fresh-Crawl Gate and Next Mechanistic Direction

Status: SUCCESS (experiment execution), **STOP (all candidate architecture promotions)**

- Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/38544398952
- Pre-registration: `research/H6_SINGLE_RUN_ARCHITECTURE_CONTRACT.md`
- Frozen R21 SHA256: `e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54`
- Model weights, thresholds, production runtime, release: unchanged.

## New evaluation data (now **consumed** for architectural evaluation)

- CC-MAIN-2026-08: 257 new samples after 2 exact-byte overlaps removed.
- CC-MAIN-2026-12: 261 new samples; 0 exact-byte overlaps removed.
- Total 518. Neither crawl should be treated as untouched holdout in future work.

### Exact Top-1 outcomes

| Detector or policy | CC08 /257 | CC12 /261 | Pooled /518 |
|---|---:|---:|---:|
| R21 control | 225 | 235 | **460 (88.80%)** |
| chardet 7 | 226 | 233 | 459 (88.61%) |
| charset-normalizer | 215 | 223 | 438 (84.56%) |
| defer_bom_rule | 224 | 236 | 460 |
| final_bom_only | 223 | 236 | 459 |
| unified_bom | 223 | 236 | 459 |
| full_b_always | 226 | 234 | 460 |
| unified_bom_always | 224 | 236 | 460 |

No architecture candidate passed the preregistered per-new-crawl no-regression gate. Do not promote. Do not retune to CC08/CC12.

### Original development collection

- R21 control: 987/1041.
- defer_bom_rule: 991/1041 (+4).
- full_b_always: 993/1041 (+6).
- final_bom_only/unified_bom/unified_bom_always: 975/1041 (-12).
- Effects did **not** generalize consistently to new crawls.

### Runtime (new 518 samples, nine trials, median)

- Control: 94.96 us/sample.
- defer_bom_rule: 104.15 us/sample (+9.7%).
- final_bom_only: 96.86 us/sample (+2.0%).
- unified_bom: 105.42 us/sample (+11.0%).
- full_b_always: 99.39 us/sample (+4.7%).
- unified_bom_always: 107.28 us/sample (+13.0%).

Important: `full_b_always` here **directly** performs Full-B scoring on eligible U/RH, omitting the baseline computation on those samples. This is an alternate direct full-B route rather than `baseline + full-B`; its timing **is not directly comparable** to H5's `full-B always` accounting without specifying computation shape. Accuracy comparisons remain meaningful because each policy is tested against the same input bytes.

## Primary next issue: CJK and RH, not UTF-8-SIG

### Ground-truth CJK family

| | R21 | chardet 7 | charset-normalizer |
|---|---:|---:|---:|
| CC08 (41 CJK) | 27 | 37 | 41 |
| CC12 (46 CJK) | 36 | 39 | 45 |
| Combined (87 CJK) | **63/87 (72.4%)** | 76/87 (87.4%) | 86/87 (98.9%) |

### Ground-truth single-byte Windows family

- CC08: R21 36/44, chardet 44/44, charset-normalizer 43/44.
- CC12: R21 50/52, chardet 52/52, charset-normalizer 52/52.
- Pooled: R21 86/96, chardet 96/96, charset-normalizer 95/96.

### RH path (regardless of truth family)

- CC08: R21 129/156, chardet 151/156, charset-normalizer 129/156.
- CC12: R21 144/156, chardet 147/156, charset-normalizer 132/156.
- Pooled RH: R21 273/312, chardet 298/312, charset-normalizer 261/312.

### Other significant routes

- U: R21 148/157, chardet 139/157, charset-normalizer 135/157.
- RL: R21 38/47, chardet 20/47, charset-normalizer 42/47.

Therefore, the pooled near-tie with chardet conceals a large CJK/RH deficit offset by performance elsewhere. Do not use only pooled accuracy to decide architecture.

### BOM annotation consistency

Among new data:
- 72 physical UTF-8 BOM samples.
- 62 annotated UTF-8-SIG and 10 annotated UTF-8.
- Dataset label assignment consults BOM and declared HTTP/body charsets, so exact-label and declared encoding evidence must be reported separately. No unconditional rewrite to UTF-8-SIG is justified.

## Architectural decision

1. **STOP** H6 BOM redesign and unconditional Full-B promotion: no stable gains.
2. **Keep** R21 frozen unchanged; no release or main update.
3. **Next one-shot mechanism study**, only after explicit preregistration, should trace **CJK and high-byte Windows errors**, not parameters:
   - Attribute whether truth was missing from rank/route, present but suppressed by postprocessing, or lost by family confusion.
   - Separate evidence channels: HTTP metadata (evaluation context only), in-band HTML/XML declarations (runtime-available bytes), BOM, byte-grammar feasibility, model raw scores and downstream changes.
   - Compare declared-encoding-aware and declaration-blind cohorts; charset-normalizer may utilize in-band declarations and the dataset ground truth may derive from the same evidence, which creates a measurement confound to quantify.
   - Test only simple specification-grounded options, not per-label exception tables or threshold sweeps.
   - Measure on the same rows across CodecCat, chardet 7, charset-normalizer, with route and true-family breakdown.
   - Reserve entirely unseen later crawls for eventual architecture validation; no reuse of H6 crawls as untouched holdout.
4. Define target behavior: pure statistical detection versus document-aware detection. In-band encoding declarations, if used, must have explicit precedence, validation and failure handling, not become a silent label-leakage optimization.

## Decision criterion for future promotion

Require improvements on independent collections, no consistent CJK/WIN regression and no unacceptable route-level degradation; maintain significant speed benefit compared to both competitors under equivalent measurement. Sample count and confidence limits matter; do not infer production-grade stability from 518 rows alone.
