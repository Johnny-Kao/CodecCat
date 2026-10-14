# H6 One-shot Architecture A/B — Pre-registration

## Purpose
Test whether replacing contradictory early BOM-based UTF-8/UTF-8-SIG overrides with an explicit single final decision owner improves cross-crawl accuracy, and whether spending compute on always-on Full-B helps. No numerical threshold or feature retuning.

## Immutable inputs
- R21 model artifact, SHA256 e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54.
- Familiar development data: CC-MAIN-2026-39, 34, 21, 17.
- New evaluation data, not previously used in CodecCat H1-H5: CC-MAIN-2026-12 and CC-MAIN-2026-08 (both published Common Crawl indexes).
- Excluded: CC-MAIN-2026-25 / R22 and CC-MAIN-2026-30.
- Remove new-evaluation exact-byte overlaps with any familiar development sample, then between newly evaluated crawls.
- Same byte samples and source labels for CodecCat, chardet 7, charset-normalizer.
- Ground-truth labels remain unchanged. Report separately whether UTF-8/SIG labels agree at family level, *not* as the headline accuracy.

## Fixed candidates — no iterative search
1. control: exact frozen Runtime.rank;
2. final_bom_only: after existing inference, if prediction is UTF-8/SIG pair, choose UTF-8-SIG iff physical UTF-8 BOM exists, otherwise UTF-8; no other encoding changes;
3. defer_bom_rule: early _rule_rerank skips its BOM-specific forced top-rank move; all remaining downstream decisions unchanged;
4. unified_bom: defer_bom_rule + final_bom_only;
5. full_b_always: always run existing frozen full-B scorer for eligible U/RH routes, leave others untouched;
6. unified_bom_always: unified_bom + full_b_always.

All use existing model weights, features, calibrators, specialist thresholds, and R12. No new heuristic feature or numeric threshold.

## Measures
- Exact Top-1 hits, per-crawl and pooled; net beneficial/harmful transitions versus frozen control.
- UTF-8/SIG family-level agreement as diagnostic, not a replacement primary metric.
- Ground-truth UTF-8 labels appearing alongside a physical UTF-8 BOM (possible label-contract conflict).
- Route-level and truth-family-level regressions.
- chardet 7, charset-normalizer exact Top-1 on identical samples.
- End-to-end runtime per sample, same run with alternating policy measurement order and repeated trials.
- Explicit pass/fail gates. Do not promote based only on pooled improvements; never use new-evaluation failures to retune candidates.

## Integrity checks
- Four-route synthetic smoke before any network collection.
- Every candidate must return one of the known model labels and a nonempty ranked list.
- Exact frozen Runtime.rank as baseline; all calculations are isolated to experiment script.
- Stop if model checksum mismatches; fail the run on assertion/error.
- Keep R21 and production untouched.
