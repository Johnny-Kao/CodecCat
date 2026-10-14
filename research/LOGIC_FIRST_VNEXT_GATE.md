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
