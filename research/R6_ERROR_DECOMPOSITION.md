# R6 Cross-Crawl Error Decomposition — 2026-10-06

## Scope

R6 asks why the single release-model path generalizes much worse than the canonical 4-fold development benchmark.

Analysis data:
- `CC-MAIN-2026-34`
- already reclassified as development data after R5.0
- 422 high-confidence rows in the formal R6 collection

Reserved evidence:
- `CC-MAIN-2026-30` was **not used** in R6 and remains untouched release-gate evidence.

Formal R6 run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37354983711

Targeted R6.1 route/BOM run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37355787420

## End-to-end decomposition

Top-1 accuracy by stage:

| Stage | Hits | Top-1 |
| --- | ---: | ---: |
| raw route scorer | 358/422 | 84.83% |
| rule rerank | 355/422 | 84.12% |
| candidate calibrator viewed alone | 362/422 | 85.78% |
| current hybrid decision | 357/422 | 84.60% |
| triad specialist | 365/422 | 86.49% |
| UTF-8 / UTF-8-SIG specialist | 369/422 | 87.44% |
| UTF-8 / GB18030 specialist | 369/422 | 87.44% |
| final guard | **371/422** | **87.91%** |

Raw scorer truth retrieval:
- Top-1: **84.83%**
- Top-3: **93.13%**
- Top-5: **95.97%**

Interpretation:
- the base representation is not globally broken;
- most truths remain near the top of the raw ranking;
- the accepted specialist stack has positive aggregate value;
- broad replacement of the scorer or downstream stack is not justified.

## Stage help / harm

Observed transition counts on the development crawl:

- rule: 8 help / 11 harm
- candidate calibration: 15 help / 8 harm
- hybrid selection: 6 help / 11 harm
- triad specialist: 9 help / 1 harm
- UTF-8 / UTF-8-SIG specialist: 6 help / 2 harm
- UTF-8 / GB18030 specialist: 2 help / 2 harm
- final replacement-rate guard: 2 help / 0 observed harm

The rule/hybrid layer deserves later cleanup, but it is not the dominant cross-crawl loss.

## Route decomposition

| Route | n | CodecCat | charset-normalizer | chardet 7 |
| --- | ---: | ---: | ---: | ---: |
| U | 120 | **94.17%** | 93.33% | 90.83% |
| RH | 246 | **89.84%** | 88.62% | 97.97% |
| RL | 54 | **68.52%** | 94.44% | 98.15% |
| N | 2 | 0% | 50% | 50% |

The central R6 finding is therefore **RL**.

R6.1 shows that this is primarily a representation/scorer problem rather than a downstream problem:

| RL stage | Hits | Top-1 |
| --- | ---: | ---: |
| raw | 35/54 | 64.81% |
| rule | 35/54 | 64.81% |
| calibrated | 37/54 | 68.52% |
| hybrid | 37/54 | 68.52% |
| triad | 37/54 | 68.52% |
| UTF-8/SIG | 37/54 | 68.52% |
| UTF-8/GB | 37/54 | 68.52% |
| final | 37/54 | 68.52% |

Existing downstream logic recovers only two RL cases. It cannot explain the ~26 percentage-point gap to charset-normalizer.

## Encoding-family decomposition

| Truth family | n | CodecCat | charset-normalizer | chardet 7 |
| --- | ---: | ---: | ---: | ---: |
| UTF | 229 | **92.14%** | 83.84% | 92.58% |
| CJK | 99 | 86.87% | 98.99% | 100% |
| single-byte Windows | 73 | 84.93% | 100% | 100% |
| single-byte ISO | 20 | 55.00% | 90.00% | 95.00% |
| KOI8 | 1 | 100% | 100% | 100% |

CodecCat is already strong on the broad UTF family. The remaining gap is concentrated in sparse legacy/CJK discrimination.

## UTF-8 vs UTF-8-SIG semantics

Largest CodecCat-vs-charset-normalizer gap confusion:
- truth `utf-8` -> CodecCat `utf-8-sig`: **11 rows**

R6.1 BOM diagnostic:
- BOM-bearing rows: 44
- ground truth `utf-8`: 12
- ground truth `utf-8-sig`: 32
- final predictions `utf-8-sig`: 41
- final predictions `utf-8`: 3
- final correct: 35/44

No-BOM rows:
- 378 total
- ground truth `utf-8`: 184
- ground truth `utf-8-sig`: 0
- final `utf-8-sig` predictions: only 2

Conclusion:
- this is partly a **label/API semantics issue** rather than a clean detector failure;
- BOM-bearing UTF-8 is labeled inconsistently between `utf-8` and `utf-8-sig` in the development ground truth;
- do not introduce a benchmark-specific mapping merely to recover points;
- public CodecCat semantics for physical BOM vs canonical encoding identity should be specified separately before changing behavior.

## R6 decision

Accepted conclusions:

1. **Do not restart performance optimization.**
2. **Do not replace the global 518-dim scorer representation yet.**
3. **Do not patch UTF-8/SIG based only on benchmark labels.**
4. **Focus the next accuracy experiment on RL only.**
5. The first RL hypothesis is sampling information loss: H/M/T-768 may be too small for inputs with <=2% high bytes, because sparse discriminative legacy bytes can be missed.
6. Any RL mechanism must remain bounded and route-local so that the current speed architecture is preserved.
7. `CC-MAIN-2026-30` remains untouched.

## R7 gate

R7 starts with one bundled development-only sampling tournament:
- H/M/T 768 baseline
- H/M/T 1536
- H/M/T 3072
- prefix 4096
- H/M/T 4096

Selection criteria:
- materially improve RL Top-1 on development data;
- do not reduce Top-3/Top-5 retrieval;
- only then reconstruct downstream behavior;
- only after a candidate is frozen should a new untouched crawl be used for external validation.
