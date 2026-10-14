# H7 Cross-domain Failure Attribution and Architecture Pre-registration

Status: DESIGN ONLY — no H7 validation run, promotion, training or release.
Baseline: immutable R21 artifact SHA256 `e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54`.
Source evidence: H5 run 38541733992; H6 run 38544398952 and its full study job log.

## Reassessment of priority

On H6 CC08+CC12 (n=518), R21 is 460/518 vs chardet 7 459/518 and charset-normalizer 438/518; this is NOT evidence of stable superiority.

* RH: R21 273/312 vs chardet 7 298/312 (25-hit deficit).
* Truth CJK: R21 63/87 vs chardet 7 76/87 vs charset-normalizer 86/87.
* Truth Windows single-byte: R21 86/96 vs chardet 7 96/96.
* U: R21 148/157 vs chardet 7 139/157.
* RL: R21 38/47 vs charset-normalizer 42/47.

The RH and CJK deficits are separate marginal distributions. Their overlap has NOT yet been established. Do not assume a particular encoding causes all RH failures.

H6 BOM changes did not pass per-crawl no-regression. Full-B always has one beneficial and one harmful transition on new data. The H6 full-B candidate bypasses baseline work rather than measuring incremental full-B cost. H6 monkeypatch overhead makes timing comparisons unsuitable as production estimates.

## Study question 1: Is the failure before, inside or after the route-local scorer?

Produce a *paired sample-level provenance table*, not only independent marginal summaries:
- crawl, fixed digest, label provenance (HTTP header, BOM, meta or detection consensus), exact truth label, label conflict flags, content length and sampled length;
- route and strict UTF-8 status, high-byte and NUL ratios, sample truncation, family;
- R21 raw top-3 and margins, after rule, calibrated, triad, pair specialists, RL specialist and final; escalation decision and full-B effect;
- frozen R21, chardet 7, charset-normalizer Top-1 predictions; exact-match and decode-compatible diagnostic labels;
- jointly grouped crawl × route × truth family × label provenance, plus error transitions;
- report failure concentration, confidence intervals where meaningful, and disagreements involving UTF-8 BOM/UTF-8-SIG.

Distinguish actual encoding uncertainty from weak or contradictory benchmark labels. Ground truth is immutable. Record exact-byte duplicates AND conflicts across different labels; do not silently dedupe away conflicts.

## Study question 2: Does the 768-byte representation discard useful discriminative evidence?

R21 baseline uses 256 byte histogram bins, 256 adjacent-byte *sum* bins (with uint8 modular arithmetic), and six aggregate quantities on HMT768 (first, middle, last 256 bytes). Full-B adds difference/XOR histograms and high-byte bands. Byte-pair sum is collision-heavy and erases ordered adjacency information. Feature sufficiency is a plausible hypothesis, NOT yet an established cause.

Predefine the following *diagnostic candidates*; do not retrain R21:
- **C0 frozen control**: byte-identical `Runtime.rank`.
- **C1 evidence-only CJK adjudication**: preserve existing final decision unless the frozen top-ranked pair belongs to competing CJK encodings; compare per-encoding decode validity and script-level character plausibility using the same bytes, with a fully specified deterministic ordering and no fitted cutoffs. Log abstentions and every changed decision.
- **C2 evidence-only cross-window adjudication**: where there are competing encodings in the frozen top ranks, require decision consistency across fixed head/middle/tail slices using the same frozen route models and existing feature implementations. Prescribe a fixed aggregation rule *before evaluation*; never fit margins to CC08/CC12.
- **C3 C1+C2**: composition gate, included only if both independent components have executable deterministic specifications before running.

These proposals may not be meaningful when the true encoding is absent from a candidate's original top set; count those ceiling cases separately. Decode validity alone cannot reliably distinguish valid single-byte encodings. Script plausibility can impose linguistic bias; report impact by language and family. Avoid making heavy probes unconditional.

Before implementing candidates, complete the observational attribution audit and check for any label or sampling leakage; replace or discard any candidate whose mechanism is not justified. No evaluation-dependent threshold editing.

## One-run H7 validation contract

1. First complete offline synthetic smoke with all four routes, BOM variants, truncated multibyte tails, ambiguous single-byte and multibyte examples, nonempty ranks, full label membership, stable frozen baseline, checksum and deterministic outputs. Run compile/import/unit checks *before* starting Common Crawl downloads.
2. Freeze exact candidate specification, script and dependency versions in staging before running evaluation.
3. Cache one deterministic collected-byte manifest and evaluate all candidates and both benchmark detectors on the identical samples. Never use CC-MAIN-2026-30; CC25/R22 only historical diagnostics and NOT tuning.
4. CC08/CC12 and CC17/21/34/39 are **all development/exploratory** after H6. If adding prospective independent crawls, log their identifiers and collection design in the preregistration before first outcome inspection. A used dataset is never described as untouched again.
5. Emit machine-readable per-sample predictions and provenance, confusion matrices, route × family cross-tabs, per-crawl accuracy, beneficial/harmful/neutral transitions, escalation counts, median/p95 runtime and peak RSS memory; time pure inference separately from network collection and diagnostic instrumentation; benchmark on identical warm inputs with counterbalanced order.
6. Acceptance: reject any candidate with an unexplained per-crawl regression, label-contract violation, non-determinism, or loss of previously strong UTF family performance. Pooled improvement alone never promotes architecture. All H7 results are research-only; human approval is required for production edits, retraining, or release.
7. Upload complete JSON/CSV artifacts with retention sufficient for audit and print the actual GitHub Actions run URL after dispatch.

## Immediate implementation sequence

A. Extract the H6/H5 paired failure distributions from source-run outputs; existing H6 results summarize aggregates and do not contain per-sample prediction traces. A new trace-producing research script is necessary.
B. Check collector provenance and normalization code; validate labels and byte sampling before designing a policy.
C. Implement C1/C2 only if steps A/B support the proposed mechanisms; otherwise execute a diagnostic-only H7 run rather than speculative A/B.
D. Keep all writes on `staging/initial-package`; do not modify main, R21 artifact, release metadata, or PyPI.
