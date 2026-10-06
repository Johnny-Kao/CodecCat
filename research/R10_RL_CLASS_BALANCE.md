# R10 RL Class-Balance Tournament — 2026-10-06

Authoritative paired run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37399050384

## Hypothesis

R9 showed that the five residual RL failures are genuine retrieval failures: their truths sit at raw ranks 10-16 rather than just being misordered inside the top candidates. Because several RL truth classes have sparse real-network support, R10 tested whether class weighting could recover retrieval without changing runtime features or architecture.

## Arms

All arms used:
- locked HMT768 / 518-dim features;
- identical reconstructed development rows;
- identical fold assignment;
- identical legacy rows;
- complete downstream refit per arm;
- no use of CC-MAIN-2026-30.

Arms:
- baseline
- RL sqrt-balanced sample weighting
- RL fully-balanced sample weighting

Only the RL route-model training weights changed.

## Paired results

| Arm | Overall | RL final | RL raw Top-1 | RL raw Top-3 |
| --- | ---: | ---: | ---: | ---: |
| baseline | **365/417 = 87.53%** | **35/40 = 87.50%** | **65.0%** | **87.5%** |
| sqrt-balanced | 365/417 = 87.53% | 35/40 = 87.50% | 57.5% | 87.5% |
| fully-balanced | 364/417 = 87.29% | 35/40 = 87.50% | 57.5% | 87.5% |

Paired delta:
- sqrt-balanced: 0 overall hits, 0 RL hits, 0 Top-3 gain
- fully-balanced: -1 overall hit, 0 RL hits, 0 Top-3 gain

## Decision

**Reject simple RL class balancing.**

Conclusions:
1. The sparse-class problem is not fixed by generic inverse-frequency weighting.
2. Weighting moves raw Top-1 in the wrong direction while leaving Top-3 retrieval unchanged.
3. The five hard RL failures are likely missing class-discriminative evidence rather than a simple prior imbalance.
4. Do not sweep more class-weight strengths; the mechanism has no positive directional evidence.
5. Keep production HMT768 / 518-dim runtime unchanged.
6. Do not spend untouched holdout evidence on R10.

## R11 direction

The next justified experiment is **RL sparse-class discriminative modeling**, targeting the hard classes identified in R9 rather than all RL classes uniformly.

First gate:
- characterize the five hard retrieval failures against candidate-specific byte validity / structural constraints;
- test whether encoding-specific validity statistics can promote the correct class without broad heuristic growth.

Candidate signals should be computed only for a small shortlisted set and only on RL:
- strict/replace decode validity;
- replacement/error rate;
- byte-pair legality where cheap;
- class-specific structural evidence for Big5 / EUC-KR / CP1251 / UTF-8.

Do not create a general heuristic stack. Any specialist must be small, bounded, and validated by paired cross-fold evidence before external validation.

CC-MAIN-2026-30 remains untouched.
