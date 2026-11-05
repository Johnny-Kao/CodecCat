# CodecCat performance benchmarks

This document records measurements of **CodecCat Native CCM2**, not the
currently published v0.1.0a2 Python Alpha. Native `0.1.0a3` remains
unreleased as of this revision.

**Source of truth:** [P0–P2 optimization results (2026-11-05)](../native-prototype/P0_P2_OPTIMIZATION_RESULTS_2026-11-05.md);
[CI resource comparison #42052594947](https://github.com/Johnny-Kao/CodecCat/actions/runs/42052594947);
[Native API regression CI #42052538603](https://github.com/Johnny-Kao/CodecCat/actions/runs/42052538603).
Raw benchmarking code:
[`bench_inference.py`](../native-prototype/bench_inference.py) and
[`resource_comparison.py`](../native-prototype/resource_comparison.py).

## 1. Optimized Native vs prior Alpha (fixed 4 KB input)

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

## 2. Four detectors on the same 304 input bytes

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

## Reproduction and measurement boundaries

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
