# Overall stratified benchmark — measured results (2026-11-07)

**Result:** [CI #42263362574](https://github.com/Johnny-Kao/CodecCat/actions/runs/42263362574) PASS. Actual released `codeccat==0.1.0`, `chardet==7.7.0`, `charset-normalizer==3.5.2`, Ubuntu 24.04/Python 3.12. The selection rules were [committed before scoring](OVERALL_BENCHMARK_PROTOCOL_2026-11-07.md). The frozen model SHA256 matched the released reference. No historical training-source exact/prefix/sampled-block overlap alerts were triggered.

## Correct Unicode output

| Cohort | Inputs | Document groups | CodecCat | chardet | charset-normalizer |
|---|---:|---:|---:|---:|---:|
| Natural UTF-8 (gnome-terminal .po) | 100 | 100 | **100/100** | **100/100** | 99/100 |
| Western and European legacy synthetic recodings | 178 | 16 | 135/178 (75.84%) | **178/178 (100%)** | 140/178 (78.65%) |
| CJK legacy synthetic recodings | 70 | 4 | **70/70** | **70/70** | **70/70** |
| Real CPython CJK legacy (previously examined) | 15 | 15 | 11/15 (73.33%) | **14/15 (93.33%)** | 11/15 (73.33%) |

**Silent incorrect Unicode output:** CodecCat 43/178 in Western synthetic, 4/15 in original CPython CJK; chardet 0/178 and 1/15 respectively; charset-normalizer 38/178 and 2/15 (plus 2 CPython abstentions). These represent incorrect, nonempty decoded text, not simply differing encoding aliases.

The source has one fresh GNOME project and a previously studied CPython fixture source. It is neither a random cross-web sample nor a representative production mix. The synthetic cohorts are deliberately correlated re-encodings of original language documents. A separate 15 UTF-8 CPython reference files were **not** included in the natural UTF-8 pool, avoiding repeated pairs in the overall breakdown. Do not pool counts into a global production accuracy number.

## Western legacy by codec

| Expected codec | N | CodecCat correct | chardet correct | charset-normalizer correct |
|---|---:|---:|---:|---:|
| cp1250 | 24 | 19 | 24 | 24 |
| cp1251 | 16 | 16 | 16 | 16 |
| cp1252 | 72 | **43** | **72** | 53 |
| cp1254 | 8 | 3 | 8 | 5 |
| iso8859-15 | 50 | 46 | 50 | 35 |
| iso8859-7 | 8 | 8 | 8 | 7 |

This identifies a reproducible **cp1252/Western misclassification problem** for later research. The 29/72 incorrect cp1252 cases dominate CodecCat's errors in this cohort. Existing evidence explicitly warns against decreasing UTF-8 rescue thresholds as a general fix: this benchmark is not an authorization to retune heuristics.

## Computational measurements (same 363 bytes, separated detector processes)

| Metric | CodecCat | chardet | charset-normalizer |
|---|---:|---:|---:|
| p50 per call | **0.5176 ms** | 1.0514 ms | 1.1284 ms |
| p95 per call | **1.4154 ms** | 2.3875 ms | 2.9781 ms |
| Sum process CPU | **280.762 ms** | 436.780 ms | 527.697 ms |
| Linux process peak RSS | 57.73 MiB | 55.79 MiB | **32.87 MiB** |

CodecCat achieves lower latency and CPU on this workload, but not the lowest RAM. Timing is a **single CI run**; differences are descriptive rather than statistically quantified. APIs perform different work internally, but measured calls include text-recovery actions for competitors.

## Descriptive balance and scope limits

An **equal weight per four cohorts** descriptive accuracy is CodecCat 87.29%, chardet 98.33%, charset-normalizer 87.75%. It is **not** general real-world accuracy, nor a benchmark-weighted model-selection mandate. The prior 843-case 97.98% contains training overlap and remains a separate development-regression score.

There were 30 short/low-non-ASCII synthetic chunks and 78 unencodable synthetic chunks skipped under fixed structural rules, and 2 locale-codec combinations with no eligible chunks. All 363 eligible cases passed the overlap screen.

**Decision:** maintain Stable 0.1.0 unchanged. Results substantiate the engineering tradeoff (speed vs silent legacy decoding errors) but cannot support a model promotion. Obtain multiple truly independent Western legacy source families and report source-grouped uncertainty before evaluating changes.

All reported source paths, lengths and hashes are metadata; raw training documents, gettext content and models are not uploaded.
