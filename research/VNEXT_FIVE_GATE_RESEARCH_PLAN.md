# CodecCat vNext — Five-batch bounded architecture decision plan

**Budget:** no more than FIVE consolidated research batches to select an architecture direction. One GitHub Actions runner job per batch; each runner must compare all predeclared candidates in the SAME process and on identical paired bytes. This research limit is distinct from later routine CI and release packaging tests. Do not open serial H experiments.

## Gate 1 — Evidence ordering and runtime trade-offs [NOW]
Predeclare R21, H14, structural late veto, strict decoder late veto, Japanese SJIS/EUC contrast, H14+strict, old early structural prescore, chardet 7.6.0 and charset-normalizer 3.5.2. All evaluated on same previously exposed development crawls: CC-MAIN-2024-46, 2025-13, 2025-51, 2026-08, 2026-12, 2026-39. Report paired benefit/harm, by crawl/family/route and full-pipeline timing. Declare labels come from WARC metadata; accuracy here is *exploratory*. Avoid 2026-25 and 2026-30. No model retraining.
**Go/no-go:** prune any rule with material harm or uncompetitive runtime; choose at most TWO evidence designs for learned ranking or conclude that pure rules are insufficient.

## Gate 2 — Unified parameter learning and topology
Freeze feature schema v2 from winning evidence design(s); train cost-aware ranking/calibration versus existing 518-d baseline. Source-/domain-disjoint development splits, label provenance checks, multiple model designs and ablations in a SINGLE runner. Same logic functions at training/inference. Bundle records schema ID. Compare data-size sensitivity and training resource needs. No hidden acceptance set inspection.
**Go/no-go:** one trainable scheme with meaningful out-of-source gain and defined memory/CPU budget, or stop.

## Gate 3 — Robustness and model choice
Frozen candidates only, cross-source and cross-year drift, encoding/family guardrails, short/truncated bytes, binary-adjacent input, adversarial tests, p50/p95 CPU and per-process RSS. One runner with staged jobs prohibited; use one job to amortize setup. Choose ONE final candidate and freeze all thresholds.

## Gate 4 — Truly independent acceptance
One locked, verified-label test set never used to design/tune/select candidate. Compare latest chardet, latest charset-normalizer, frozen R21, and candidate on SAME bytes, source-disjoint. Report paired bootstrap CIs and family false-rejects, plus speed/RSS. If genuine verified external labels cannot be assembled, classify gate as BLOCKED rather than promoting HTML-declared labels to truth.
**Pass:** candidate beats BOTH current competitors overall on verified holdout with credible paired margin and without unacceptable family regressions.

## Gate 5 — Final convergence
Only if Gate 4 passes: deterministic reproduce, cross-Python/CPU where available, model bundle compatibility and release candidate QC. If Gate 4 fails, use this LAST research slot for one diagnostic to select direction (not repeated threshold tuning), then stop. No production push without explicit user authorization.

## Cost controls
- ONE Actions workflow run per phase; no multi-matrix jobs or paid runners. Data cached/downloaded once per run; one summary JSON and sample traces as artifact.
- No unrelated auto-push test workflows during staging research. Prefer the existing runner and dependency pins.
- Freeze each batch's candidate list before running; modifications after seeing output consume a new batch. Hard cap of five research batches.
- H1–H14 are research-exposed; no claim of independent holdout. R21 immutable NPZ SHA256: e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54.
- Dependency/algorithm literature: chardet 7.6 deterministic multi-stage candidate/byte screening plus n-gram profiles; charset-normalizer 3.5 uses decoded-text mess and coherence. Reimplement ideas independently, do not copy source.

## Gate 1 sign-off — completed (2026-10-15)

Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/38657777163
Artifact: vnext-gate1-paired-tournament (report.json and paired_trace.jsonl).
6 previously exposed research crawls, 1,550 SHA-deduplicated samples. Exact competitor versions: chardet 7.6.0 and charset-normalizer 3.5.2.

| Fixed policy | Correct/1550 | Mean-throughput-like median ns/sample |
|---|---:|---:|
| R21 | 1412 | 152312 |
| H14 | 1435 | 179549 |
| Structural late | 1426 | 233602 |
| Strict decoding late | 1426 | 159646 |
| Japanese contrast | 1442 | 203580 |
| H14 + strict late | 1446 | 193770 |
| Structural prescore (prior H2 architecture) | 1348 | 831394 |
| chardet 7.6 | 1484 | 1116381 |
| charset-normalizer 3.5.2 | 1384 | 922898 |

**Key decision:** NEVER promote structural mask into pre-score hard exclusion without an independent case for safety and ranking interactions: 71 previously correct cases were made wrong, 7 fixed, net -64. Logic-first means deterministic structural evidence is computed and validated *before model-policy design*, NOT necessarily applied as a hard mask before raw scoring.

Late H14+strict fixes 34 relative to R21 with zero observed harmful transitions; it remains 38 hits behind chardet 7.6.0 (2.45 percentage points) and is only 5.76x faster on this paired throughput benchmark, below the earlier desired >=10x newest-chardet target. Do not claim superiority or release.

### Error budget / targeted Gate 2 directions
- R21 Top-3 contains label in 1488/1550 (96.0%); 62 labels absent, so Top-3-only reranking cannot meet newest chardet's present accuracy alone.
- Single-byte ISO: 52/83 correct R21 and H14+strict, versus 73/83 chardet. In particular ISO-8859-2 only 3/12 vs 11/12. **Ordered directional byte-pair / language-statistics features** take priority.
- Japanese: Shift-JIS 9/33 R21 to 32/33 H14+strict (chardet 32/33); EUC-JP remains 12/20 while chardet is 20/20. Include **dual-Japanese structure and language evidence** with abstention, not unconditional override.
- GB18030: 79/88 H14+strict versus 86/88 chardet. Include **GB/CJK candidate evidence**.
- CP1251: 218/222 H14+strict versus 221/222 chardet. Need test whether additional ordered bigram evidence helps.
- UTF: R21 753/798, H14+strict 755/798 versus chardet 746/798, so preserve UTF strength.
- Worst route gap: RL 150/178 R21 vs chardet 172/178. Do not globally run expensive RH-only logic on every input.
- Six-crawl sample labels derive from document declarations, agreement and strict decoding. Competing library may itself read those declarations; explicitly stratify visible declaration vs no declaration in Gate 2 and never equate declared labels with independent human verification.

### Revised locked Gate 2 A/B scope
In ONE runner, collect and cache source metadata and sampled bytes ONCE, retain group IDs by WARC path and host, and compare:
1. Existing frozen R21 and late H14+strict as controls.
2. Re-trained 518-d base only (tests whether more domain data, not feature architecture, helps).
3. Versioned structural features + learned scorer, with evidence flags and ambiguity intact.
4. Ordered, noncommutative byte-pair directional features + learned scorer (specific ISO/RL hypothesis).
5. Combined structural + ordered byte-pair features + learned scorer and selective policy.
No new thresholds after seeing results. Fit fixed-configuration models with source-disjoint folds and report per-label/route/source, plus throughput and the model/schema bytes. Preserve raw-data caching in the artifact subject to size limits; make any truncation explicit, and never treat cached development bytes as future untouched holdout.
Candidate decision: if no trainable candidate beats R21+H14+strict across source-held-out evaluation, flag learned-feature direction as unproven and use Gate 3 for a fundamentally different decision topology, not tweaks to the same parameters.
