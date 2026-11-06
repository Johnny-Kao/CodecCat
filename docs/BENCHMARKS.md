# CodecCat accuracy and performance benchmarks

## Five-corpus benchmark (released Native v0.1.0a5)

[Post-release run #42158308518](https://github.com/Johnny-Kao/CodecCat/actions/runs/42158308518) installed the **published v0.1.0a5 Wheel**. The same 843 byte samples across five fixed sets were compared with chardet 7.7.0 and charset-normalizer 3.5.2 on Ubuntu 24.04 / Python 3.12.

| Metric | CodecCat a5 | chardet | charset-normalizer |
|---|---:|---:|---:|
| Correct Unicode text | 826/843 (97.98%) | **829/843 (98.34%)** | 698/843 (82.80%) |
| Exact encoding label | **787/843 (93.36%)** | 712/843 (84.46%) | 583/843 (69.16%) |
| Total CPU time | **660 ms** | 861 ms | 1,073 ms |
| p50 latency range | 0.78–0.89 ms | **0.46–0.75 ms** | 0.74–1.02 ms |
| p95 latency range | **1.20–1.32 ms** | 2.69–3.01 ms | 3.10–4.64 ms |
| Whole-process peak RSS | 66–69 MiB | 61–64 MiB | **40–44 MiB** |

The new version correctly decodes **190/195** previously examined Vim/Git samples (a4: 183/195). This development benchmark includes 633 samples from the model's training source; **97.98% is not an independent real-world accuracy estimate**. CPU and latency values are one hosted CI observation, not guaranteed differences.

[Same-model paired evidence and limitations](../research/UTF8_EVIDENCE_V2_2026-11-06.md). Historical a4 results remain below.

## Source-disjoint follow-up checks (2026-11-06)

These are **new source repositories**, selected and pinned before running the published `v0.1.0a4` Wheel. They are distinct from the model's documented training repository, but no exhaustive semantic near-duplicate audit has been completed.

| Source | Samples | Encodings represented | CodecCat correct text | chardet | charset-normalizer |
|---|---:|---|---:|---:|---:|
| GNOME GTK translation files | 328 | UTF-8 only | **296 (90.24%)** | **328 (100%)** | 326 (99.39%) |
| uchardet original test files | 102 | 35 Python-supported encodings | **94 (92.16%)** | **96 (94.12%)** | 90 (88.24%) |

[GTK five-fold CI #42153394720](https://github.com/Johnny-Kao/CodecCat/actions/runs/42153394720) and [multi-encoding five-fold CI #42153686436](https://github.com/Johnny-Kao/CodecCat/actions/runs/42153686436) both PASS as benchmark executions. The five groups are **document-level partitions**, not five independently collected sources. Files from the same document never cross folds. The multi-encoding source includes short original test files and correlated prefix windows; `viscii` and `euc-tw` were excluded because standard Python lacks those codecs. No model tuning was performed based on these held-out results.

**These checks do not pass a Stable-release quality criterion:** GTK UTF-8 underperforms both competitors; on the multi-encoding source CodecCat is below chardet. A separate external-source regression test passing only proves the test ran, not product accuracy equivalence.

## Frozen release-model reproducibility

Future **release-platform matrix builds** and **root-wheel smoke builds** now retrieve a frozen model/reference archive from GitHub Release `v0.1.0a4` rather than retraining on every run. The archive SHA256 is pinned to `e3dd654817fb0e3324910b1a39af9bc19185400582d5a066c533d7cc5dd81c8c`, and the default model SHA256 to `16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b`. The workflow fails closed when either differs.

[Six-platform frozen reference verification #42153391865](https://github.com/Johnny-Kao/CodecCat/actions/runs/42153391865) and [root-wheel frozen reference verification #42153902552](https://github.com/Johnny-Kao/CodecCat/actions/runs/42153902552) both PASS. This ensures repeatable **model bytes in release builds**, not bit-for-bit deterministic model training or identical Wheel ZIP timestamps. Offline experimentation may still retrain independently; its outputs cannot silently replace the release model.

[Full model-freeze and external-holdout report](../research/FROZEN_MODEL_AND_EXTERNAL_HOLDOUT_2026-11-06.md).

## Five-corpus benchmark (released Native v0.1.0a4)

The latest measurement installed the **actual GitHub Release v0.1.0a4 Wheel**, not a development build. [Post-release five-corpus CI #42151051728](https://github.com/Johnny-Kao/CodecCat/actions/runs/42151051728) completed on 843 fixed byte inputs, comparing correct Unicode decoding, encoding labels, CPU, p50/p95 and RSS with chardet 7.7.0 and charset-normalizer 3.5.2.

[Paired release validation #42151536276](https://github.com/Johnny-Kao/CodecCat/actions/runs/42151536276) installed **both published v0.1.0a3 and v0.1.0a4** and asserted identical embedded model SHA256 (`16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b`). On the same 285 previously unexamined GNOME GLib translation windows, correctly decoded Unicode text improved **227 → 271**, with **44 helps and 0 observed harms**. This attributes the paired change to decoding logic rather than retraining.

The [full source-stratified report](../research/NATIVE_UTF8_FIX_2026-11-06.md) also documents the difference between provisional staging results and the shipped release. Training and prior research overlap in the 843-case test; **97.15% is not general real-world accuracy**. Hosted-runner CPU/latency values are one-run observations.

Historical stable23/v0.1.0a3 measurements remain below for provenance, not for the current release.

## Accuracy across data sources

CodecCat's native default is derived from the historical stable23 classifier;
the release CI verified behavioral parity on fixed regression inputs.
**These accuracy evaluations ran the earlier stable23 implementation** rather
than the downloaded `v0.1.0a3` Native wheel. They therefore describe the
classifier's earlier external-source behavior, not an independent blind
release-artifact acceptance test.

The primary measure here is whether the detected encoding can be strictly
decoded to the **correct Unicode text**, not whether the name of the original
encoding was recovered.

| Original data source | Evaluated | stable23 | chardet 7.7.0 | charset-normalizer 3.5.2 |
|:--|--:|--:|--:|--:|
| Vim translation strings | 159 | 155 (97.48%) | **159 (100%)** | 143 (89.94%) |
| Git translation strings | 76 | 76 (100%) | 76 (100%) | 76 (100%) |
| Tatoeba CC0 sentences, synthetically re-encoded | 560 | 524 (93.57%) | **554 (98.93%)** | 498 (88.93%) |

The three source families should **not be pooled into a single general
accuracy percentage**. The Git samples are UTF-8 only, Vim samples include
correlated windows from source catalogs, and Tatoeba's chosen byte encodings
are synthetic rather than naturally occurring frequency distributions.

Test provenance and CI:
- [Vim 159 real translation windows, CI #41238286450](https://github.com/Johnny-Kao/CodecCat/actions/runs/41238286450)
- [Git 76 real translation windows, CI #41239422354](https://github.com/Johnny-Kao/CodecCat/actions/runs/41239422354)
- [Tatoeba 560 frozen CC0-derived sentences, CI #41233855912](https://github.com/Johnny-Kao/CodecCat/actions/runs/41233855912)
- [Research report and source-contamination analysis](https://github.com/Johnny-Kao/CodecCat/blob/research/accuracy-data-gate-v2/research/REAL_BYTE_ACCURACY_EVIDENCE_2026-11-01.md)
- [Accuracy evaluation and model promotion decisions](https://github.com/Johnny-Kao/CodecCat/blob/research/accuracy-data-gate-v2/research/ACCURACY_SOLUTION_EXECUTION_2026-11-01.md)

The 3,092-item / 94.24% historical development benchmark belongs to a
**different experimental binary 1–2-gram research model**, not the packaged
Native stable23 model. It is **not** the release's accuracy figure. That
benchmark also included data used in development.

## Performance and resources

**Source of truth:** [P0–P2 optimization results (2026-11-05)](../native-prototype/P0_P2_OPTIMIZATION_RESULTS_2026-11-05.md);
[CI resource comparison #42052594947](https://github.com/Johnny-Kao/CodecCat/actions/runs/42052594947);
[Native API regression CI #42052538603](https://github.com/Johnny-Kao/CodecCat/actions/runs/42052538603).
Raw benchmarking code:
[`bench_inference.py`](../native-prototype/bench_inference.py) and
[`resource_comparison.py`](../native-prototype/resource_comparison.py).

### Optimized Native vs prior Alpha (fixed 4 KB input)

One Linux x86-64 GitHub Actions runner, separate interpreters with their
corresponding installed dependencies, original Alpha stable23 model vs the
same model exported to native CCM2 float32.

| Metric | Native CCM2 | Alpha / NumPy + scikit-learn | Relative change |
|:--|--:|--:|--:|
| 4 KB detection p50 | 0.1908 ms | 2.4429 ms | ~12.8× faster |
| 4 KB detection p95 | 0.2018 ms | 2.5649 ms | ~12.7× faster |
| Model load | 41.3 ms | 838.5 ms | ~20.3× faster |
| Peak process RSS | 54.7 MiB | 212.7 MiB | ~74% lower |
| Uncompressed model artifact | 16.64 MB | 32.48 MB | ~49% smaller |

These are **example single-run results**, not ranges, calibrated confidence
intervals, or a guarantee on other machines. The model-file size is **not**
the same as an installed virtual environment size or a process's RSS.
The benchmark's measured workload is an API call, not an isolated
mathematical kernel.

### Four detectors on the same 304 input bytes

[Resource benchmark CI #42052594947](https://github.com/Johnny-Kao/CodecCat/actions/runs/42052594947)
used one Ubuntu 24.04 x86-64 GitHub runner with **Python 3.12**,
different isolated virtual environments, common input order, and
**304 total cases** (80 short, 220 long, 4 edge cases).

| Metric | CodecCat Native CCM2 | CodecCat Alpha | chardet 7.5.0 | charset-normalizer 3.5.2 |
|:--|--:|--:|--:|--:|
| Detection wall time p50, ms | **0.4712** | 1.4649 | 0.5388 | 0.8903 |
| Detection wall time p95, ms | **1.4944** | 4.6826 | 3.6182 | 2.7654 |
| Total process CPU across calls, ms | **182.95** | 609.78 | 314.61 | 347.20 |
| Peak process RSS, MiB | 55.12 | 211.71 | 51.59 | **25.88** |
| Isolated venv disk usage, MB | 30 | 278 | 16 | **14** |
| Exact Unicode text recovered, of 304 | 294 | 294 | 286 | 249 |

Interpretation:
- Native's measured median detection latency was ~1.14× faster than
  chardet and ~1.89× faster than charset-normalizer on this **specific** mix.
- CodecCat Native was **not** the lowest-memory or smallest-installation
  option: peak RSS was 55.12 MiB versus 51.59 MiB for chardet and 25.88
  MiB for charset-normalizer.
- All implementations processed identical bytes, but the APIs are **not
  identical work units**. CodecCat detects and strictly decodes in the timed
  call. chardet returns an encoding label. charset-normalizer uses
  `from_bytes(...).best()`. For the text-recovery comparison, common
  strict decoding of competitor outputs ran **after timing**.
- These bytes are selected from a dataset previously used for development
  (`chardet/test-data`) plus fixed edge cases. **294/304 must not be
  marketed as independent 96.7% accuracy** or as performance on a new
  held-out dataset. The separate 94.243% research model is **not** the
  packaged Alpha-derived classifier.

### Reproduction and measurement boundaries

The reproducible source is the workflow
[`native-resource-benchmark.yml`](../.github/workflows/native-resource-benchmark.yml)
and the fixed corpus commit
[`b0c0d206913ba118ffea2f7d306883e7a4a76db7`](https://github.com/chardet/test-data/commit/b0c0d206913ba118ffea2f7d306883e7a4a76db7).
The runner installs separate virtual environments, prepares an identical
fixture set, warms each detector on five inputs, and measures one call at a
time with `perf_counter_ns` and `process_time_ns`. Peak process RSS is
from `resource.getrusage`; environment disk size is from `du -sm`.

The resource workflow remains a historical, **manually dispatched**
benchmark rather than a blocking release performance gate. It currently
builds through the native source subdirectory; the official release
package is built at the repository root. If you rerun it, compare tool
versions, wheel model fingerprints, and runner environment before
comparing absolute numbers. The project does not claim that independent
repeat-run distribution, confidence intervals, or other OS benchmark
results have been established.

For CPU performance assessment, prefer an untuned, source-independent
benchmark with matched API semantics and reported p50/p95 distributions
across several runs. For general encoding accuracy, use a separately held
out dataset with source-group isolation.
