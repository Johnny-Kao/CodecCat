# Logic-First vNext — reset and evidence governance

## Current scope
An isolated **structural viability evidence** layer is introduced in `src/codeccat/logic_first.py`. It does NOT overwrite R21 scores, does NOT choose an encoding, and has not been added to production Runtime. The output is `valid`, `invalid`, or `incomplete` for a byte buffer. This MVP covers UTF-8, Shift-JIS, EUC-JP, Big5 and GB18030.

Structural validity is **not** full codec decodability: valid byte-range patterns may still include unassigned or illegal mapped codepoints for a given codec/version. Strict decoder comparisons are needed before hard exclusion. Distinguish complete file versus truncated-prefix semantics.

## Evidence inventory and retest policy
- R21 frozen artifact and SHA remain immutable. Its established product benchmark is the historical comparator, not a test for new logic.
- H1–H7, H10–H13: **exploratory diagnostics**; preserve as historic evidence, never promote them to untouched holdouts.
- H8: rejected change to sampling under old weights. Its negative result does not automatically apply to the new logic-first architecture.
- H9: isolated Python/NumPy/model memory cost is a reusable runtime baseline; remeasure a final candidate.
- H14: +41 improvements on 3355 **research-exposed** samples with 19% runtime overhead; development-only proof of concept. Rerun on locked candidate after logic validation.
- vNext manifest trainer: existing 518-d baseline training contract, independent of Logic-First vNext. Do not label its passing unit tests as proof of any language-level quality gate.

## Stage order
1. **Logic Gate**: known-valid bytes never marked invalid; invalid-state transitions rejected; truncated buffers incomplete; ambiguous ASCII stays ambiguous; cross-codec differential tests and versioned reference checks; no unjustified exclusions.
2. **Evidence Feature Schema**: version structural statuses and language evidence. Production scorer must not consume a new feature schema without compatible weights.
3. **Training Gate**: one reproducible user-supplied dataset trainer with source-disjoint train/validation/test, no exact byte conflicts, fixed seeds and dependency versions; model bundle includes feature-schema ID.
4. **Evaluation Gate**: independent verified-label corpus, per-family and by-source reporting, latest chardet and charset-normalizer, speed + memory. No data leakage; acceptance corpus closed until challenger frozen.
5. **Release Gate**: only if accuracy beats both competitors with no meaningful family regressions and performance remains competitive; otherwise retain frozen R21.

## Non-goals
Do not retrain R21, modify main, change released packages, or declare that a structural validator itself improves detection accuracy. No paid runners.


## Single-batch execution rule (2026-10-15)
- One CI setup, one runner job, one seeded batch, one result artifact; never launch a standalone workflow per hypothesis or encoding family.
- Group cheap unit checks, contract checks, deterministic exhaustive spaces, property-based generated valid samples, adversarial/random inputs, negative mutations, truncation, and CPU/RSS evidence into the same execution.
- Freeze inputs/seeds and register all candidates before launching; avoid inspecting partial results to start follow-on runner jobs.
- Consolidated workflow: .github/workflows/logic-first-batch.yml. Only a deliberate workflow-file update on staging triggers the batch; workflow_dispatch is available once the workflow exists on the default branch. The prior logic-first-gate.yml auto-push trigger is retired.
- Preserve JSON with per-encoding counters as the downloadable Actions artifact, and log high-level summary. One benchmark invocation should gather multiple metrics, not one test per runner invocation.

### First batch evidence
- Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/38651401085
- Exactly 1,055,568 structural checks, including exhaustive 65,536 two-byte strings across five encodings, 50,000 generated known-valid samples per encoding, 60,000 random/adversarial strings across five encodings, mutations, truncations and 20,000 complete five-codec performance passes.
- Differential cross-check against Python strict codecs: 0 false exclusions observed. Existing logic and dataset/training tests: 11/11 passed.
- Measured script phases: 1.168 s exhaustive; 7.666 s generated; 2.371 s random/adversarial; 0.578 s five-codec throughput, for 11.783 s substantive batch execution (not including unit tests, package install, or Actions provisioning).
- Full-five-codec structural throughput 172,881 encoding evaluations/s in this run; batch process peak 32.36 MiB RSS. These are *not* production detector timings or model RSS metrics.
- This result supports byte-structure soundness on synthetic/reference-codec samples. It is NOT a general accuracy proof, an exhaustive multi-byte proof, nor evidence that new architecture beats chardet or charset-normalizer.
- Do not repeat this large batch unchanged. Next meaningful batch should include new independently verified multi-encoding datasets, schema experiments and final candidate comparisons as one predeclared tournament.
