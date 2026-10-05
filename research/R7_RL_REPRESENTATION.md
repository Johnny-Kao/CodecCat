# R7 RL Representation Research — 2026-10-06

## Trigger

R6 isolated RL as the dominant cross-crawl accuracy weakness:

- RL final: 37/54 = 68.52%
- charset-normalizer: 51/54 = 94.44%
- RL raw scorer: 35/54 = 64.81%

Existing downstream logic recovered only two RL rows, so R7 targets representation/scoring rather than another reranker.

R6 record:
`research/R6_ERROR_DECOMPOSITION.md`

## Hypothesis R7-A — H/M/T-768 loses sparse legacy-byte evidence

RL is defined by a high-byte ratio <=2% after excluding strict UTF-8 and NUL-bearing data.

On long RL inputs, only 768 H/M/T bytes are represented by the scorer. When discriminative legacy bytes are sparse, a 768-byte sample can miss them.

Candidate family:
- HMT768 baseline
- HMT1536
- HMT3072
- prefix4096
- HMT4096

Only RL changes. All other routes remain HMT768.

## Corpus-control correction

Early live Common Crawl tournament attempts produced only 34 RL rows because range availability can change between workflow runs.

Those runs are directional only.

R7 therefore freezes one expanded development-only RL corpus in GitHub Actions cache:

`codeccat-r7-ccmain202634-rl-v1`

Frozen corpus:
- source: `CC-MAIN-2026-34`
- RL rows: **62**
- Common Crawl HTTP 206 ranges: 54
- responses seen: 5,761
- no `CC-MAIN-2026-30` use

Authoritative frozen-corpus tournament:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37357086041

## Frozen-corpus raw scorer tournament

| Candidate | Top-1 | Top-3 | Top-5 | Δ Top-1 vs 768 |
| --- | ---: | ---: | ---: | ---: |
| HMT768 | 41/62 = 66.13% | 72.58% | 80.65% | baseline |
| HMT1536 | 41/62 = 66.13% | 77.42% | 82.26% | +0.00 pp |
| HMT3072 | 45/62 = 72.58% | 80.65% | 83.87% | +6.45 pp |
| prefix4096 | 42/62 = 67.74% | 72.58% | 80.65% | +1.61 pp |
| **HMT4096** | **46/62 = 74.19%** | **80.65%** | **83.87%** | **+8.06 pp** |

Interpretation:
- merely adding bytes is not sufficient: prefix4096 adds little;
- preserving head/middle/tail coverage matters;
- 1536 is insufficient;
- the useful regime appears around 3072-4096 bytes;
- HMT4096 wins Top-1 while matching HMT3072 Top-3/Top-5;
- the hypothesis that HMT768 loses sparse distributed evidence is supported.

## Acceptance boundary

The raw scorer result is **not yet a production acceptance**.

Changing RL representation changes:
- route-model fit;
- calibration score distributions;
- triad features derived from model scores;
- pair-specialist score inputs.

Therefore all downstream models must be rebuilt consistently.

A first directional full-fold rebuild was launched:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37357479010

Methodology issue discovered before accepting it:
- original cached fold state was built from 420 external rows;
- only 418 evaluable rows are retained in the cached fold fixture;
- historical 365/418 therefore cannot be used as a direct baseline for a retrain on reconstructed rows.

Controlled acceptance run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37357863157

The controlled run retrains both:
1. baseline RL HMT768;
2. candidate RL HMT4096;

using the exact same:
- reconstructed 418 external rows;
- fold assignment;
- legacy rows;
- model fitting;
- downstream calibration pipeline.

Only the paired delta is authoritative for this gate.

## Reserved evidence

`CC-MAIN-2026-30` remains untouched by R7 selection and training.

If HMT4096 passes the controlled full-pipeline gate, the next external validation must use a newly reserved crawl before production acceptance.
