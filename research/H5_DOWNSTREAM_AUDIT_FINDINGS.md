# H5 Downstream Audit — Architectural Findings

Status: diagnostic complete; no runtime or model promotion.

Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/38541733992
Job: 115777882577
R21 frozen artifact SHA256: e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54

## Scope and data

- Existing development crawls only: CC-MAIN-2026-17 (260), CC-MAIN-2026-21 (261), CC-MAIN-2026-34 (261), CC-MAIN-2026-39 (260).
- Pooled, deduplicated observations: 1,041.
- No R22, no CC30.
- No model training, parameter tuning, threshold sweep, or runtime edits.
- Four-route synthetic smoke checks completed before network collection.

## Pooled stage results

| Stage | Correct | Sample N |
|---|---:|---:|
| Baseline raw | 976 | 1,041 |
| Baseline rule | 960 | 1,041 |
| Baseline hybrid | 960 | 1,041 |
| Baseline triad gate | 958 | 1,041 |
| Baseline UTF-8/SIG pair | 967 | 1,041 |
| Baseline UTF-8/GB pair | 966 | 1,041 |
| Baseline replacement guard | 966 | 1,041 |
| Baseline + RL specialist | 967 | 1,041 |
| Full-B raw (eligible U/RH only) | 896 | 926 |
| Full-B rule (eligible U/RH only) | 872 | 926 |
| Full-B final (eligible U/RH only) | 891 | 926 |
| Frozen R21 end-to-end | 987 | 1,041 |
| Full-B always on eligible U/RH | 993 | 1,041 |

### Mechanism-level transitions

- Baseline raw -> rule: 56 changes, 19 beneficial, 35 harmful, net -16.
- Full-B raw -> rule: 43 changes, 9 beneficial, 33 harmful, net -24.
- Baseline hybrid -> triad: 17 changes, 5 beneficial, 7 harmful, net -2.
- Baseline triad -> UTF-8/SIG pair: 9 changes, 9 beneficial, 0 harmful, net +9.
- Full-B triad -> UTF-8/SIG pair: 24 changes, 21 beneficial, 3 harmful, net +18.
- Baseline UTF-8/SIG -> UTF-8/GB pair: 4 changes, 1 beneficial, 2 harmful, net -1.
- RL specialist: 4 changes, 2 beneficial, 1 harmful, net +1.
- Frozen full-B escalation (versus baseline final): 30 changes, 22 beneficial, 2 harmful; net +20.
- Full-B always versus frozen: 10 changes, 8 beneficial, 2 harmful; net +6.

### Specific high-impact rule transition

- Baseline raw -> rule: `utf-8` -> `utf-8-sig` appeared 41 times.
- Full-B raw -> rule: `utf-8` -> `utf-8-sig` appeared 37 times.
- Baseline UTF-8/SIG specialist often reverses this: 9/9 reversals were beneficial.
- Full-B UTF-8/SIG specialist reverses this: 21 beneficial and 3 harmful among 24.
- The data-labeling procedure separately handles BOM, HTTP charset and in-body declarations. Where header and body agree on `utf-8`, they can outvote a BOM-based `utf-8-sig` observation. This creates a *plausible* label-semantics collision; it must be audited as a contract issue rather than tuned away.

## Per-crawl raw -> rule harm

| Crawl | Baseline raw->rule harmful | beneficial | net | Full-B raw->rule harmful | beneficial | net |
|---|---:|---:|---:|---:|---:|---:|
| CC17 | 11 | 9 | -2 | 10 | 3 | -7 |
| CC21 | 2 | 10 | +8 | 2 | 6 | +4 |
| CC34 | 12 | 0 | -12 | 11 | 0 | -11 |
| CC39 | 10 | 0 | -10 | 10 | 0 | -10 |

## Decision

**Primary architecture issue identified: unconditional post-score rule overrides, especially BOM / UTF-8-SIG treatment, conflict with scorer preference and possibly the dataset's label semantics.**

Do NOT directly delete all downstream rules:
- CC21 benefits from raw->rule behavior;
- the UTF-8/SIG pair corrects many previous mistakes;
- all ablations were observational, not independently validated counterfactual production policies.

Do NOT tune a new margin, ratio or route threshold.

### Next action before any A/B promotion

1. Define an explicit detection target and label-equivalence policy for UTF-8 vs UTF-8-SIG, including BOM-bearing input and declared charset. Separate exact label match from practical decode-equivalence; retain both metrics.
2. Audit whether `_rule_rerank`'s BOM override and the later SIG pair are contradictory by design, and test an invariant-preserving single decision owner for BOM/SIG.
3. If the label semantics and invariant are specified, preregister **one** architecture-level A/B against immutable R21: same inputs across all three detectors, with crawl-level breakdown, beneficial/harmful transitions, latency and zero-tuning policy.
4. Validate on untouched independent data only after architecture selection. No release claim from the current dev aggregate.

No accuracy/runtime promotion made from H5.
