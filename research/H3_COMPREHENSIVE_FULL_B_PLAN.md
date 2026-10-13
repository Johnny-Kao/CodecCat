# H3 Comprehensive Full-B Necessity Study

Status: preregistered research
Release line affected: none

## Goal

Determine in one experiment whether the expensive full-B escalation can be reduced using cheap, interpretable baseline information without parameter tuning.

## Fixed constraints

- Frozen R21 artifact.
- Development crawls only: CC-MAIN-2026-39 and CC-MAIN-2026-34.
- R22 / CC-MAIN-2026-25 excluded.
- No retraining.
- No changes to C, R12, features, release thresholds, or model weights.
- No broad parameter sweep.

## Questions answered in one run

1. How often does current full-B escalation change the final result?
2. Of those changes, how many are beneficial, harmful, or neutral versus the baseline-only path?
3. Which cheap baseline signals characterize useful versus no-op escalations?
4. Can a small set of predeclared skip-gate families reduce full-B work without accuracy loss?
5. What is the same-run runtime effect?
6. Are results directionally consistent across both development crawls?

## Cheap signals

Only already-available baseline information:
- route;
- raw top1-top2 margin;
- top1 label;
- top2 label;
- top1/top2 family identity;
- same-family flag;
- strict UTF-8;
- BOM;
- ASCII-only;
- high-byte ratio;
- NUL ratio.

## Predeclared gate families

Control:
- current frozen gate.

Candidate A — strong-margin skip:
- among samples the frozen gate would escalate, skip full-B if margin is above a fixed fraction of the existing frozen threshold.
- Fractions tested: 0.50 and 0.75 only.

Candidate B — same-family strong-margin skip:
- same as A, but only when baseline top1/top2 are in the same encoding family.

Candidate C — different-family strong-margin skip:
- same as A, but only when baseline top1/top2 are in different families.

These are mechanism probes, not a search for an optimum. No additional fractions may be introduced after results are observed.

## Outputs

For every candidate:
- final accuracy;
- delta hits vs frozen control;
- full-B triggers;
- trigger reduction;
- useful full-B skips missed;
- harmful skips;
- per-crawl results;
- per-route results;
- same-run runtime.

The run must also report the raw taxonomy of current full-B escalations:
- final changed vs unchanged;
- baseline wrong -> full-B correct;
- baseline correct -> full-B wrong;
- wrong -> different wrong;
- correct -> same correct.

## Decision rule

A candidate is architecture-worthy only if:
- no development crawl loses hits;
- pooled hits do not decrease;
- full-B triggers materially decrease;
- runtime improves;
- the mechanism has a simple explanation.

If no candidate passes, do not tune more thresholds to rescue it. Keep the frozen gate and stop this optimization line.
