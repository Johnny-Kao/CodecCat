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
