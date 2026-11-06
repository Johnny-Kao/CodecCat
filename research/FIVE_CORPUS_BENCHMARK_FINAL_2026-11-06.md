# Five-corpus benchmark — CodecCat v0.1.0a3 (2026-11-06)

**Final validated run:** [GitHub Actions #42136798760](https://github.com/Johnny-Kao/CodecCat/actions/runs/42136798760) — **PASS**.

The benchmark installs the **published Native `v0.1.0a3` Wheel** and compares it with `chardet==7.7.0` and `charset-normalizer==3.5.2` on the same inputs. Python 3.12, Ubuntu 24.04 runner; each program starts in a separate process, with warm-up before timed detection.

## Summary — 843 inputs, five fixed corpora

| Metric | CodecCat Native | chardet | charset-normalizer |
|---|---:|---:|---:|
| Correctly decoded text | 805 / 843 (**95.49%**) | **829 / 843 (98.34%)** | 698 / 843 (82.80%) |
| Exact encoding label | **766 / 843 (90.87%)** | 712 / 843 (84.46%) | 583 / 843 (69.16%) |
| Total CPU time, 843 calls | **598 ms** | 732 ms | 881 ms |
| Median latency (range across 5 corpora) | 0.73–0.80 ms | **0.36–0.61 ms** | 0.63–0.84 ms |
| p95 latency (range across 5 corpora) | **1.12–1.20 ms** | 2.11–3.10 ms | 2.50–3.89 ms |
| Peak process RAM, across 5 runs | 66.2–68.5 MiB | 60.9–63.8 MiB | **40.0–43.8 MiB** |

**Conclusion:** chardet correctly decoded more text and had lower typical latency. CodecCat used less total measured CPU time and had lower p95 latency, but higher peak RAM. Charset-normalizer required the least RAM.

## Correct text by corpus

| Corpus | Cases | CodecCat | chardet | charset-normalizer |
|---|---:|---:|---:|---:|
| 1 | 185 | 173 | **182** | 150 |
| 2 | 151 | **149** | 147 | 133 |
| 3 | 164 | 155 | **163** | 143 |
| 4 | 163 | 155 | **159** | 134 |
| 5 | 180 | 173 | **178** | 138 |
| **Total** | **843** | **805** | **829** | **698** |

## What did these five corpora actually contain?

The benchmark uses **seed 20261106**, selects original labeled bytes from three source pools, deduplicates identical bytes, and groups source files before assigning them to five corpora. It is **not** five independently gathered blind datasets.

| Source pool | Cases | CodecCat correct text | chardet | charset-normalizer |
|---|---:|---:|---:|---:|
| chardet/test-data | 633 | 622 (98.26%) | 622 (98.26%) | 508 (80.25%) |
| charset-normalizer's own labeled fixtures | 15 | 14 (93.33%) | 15 (100%) | 15 (100%) |
| Vim / Git translation source files | 195 | 169 (86.67%) | 192 (98.46%) | 175 (89.74%) |

**Main concern:** the `chardet/test-data` pool appears in CodecCat's **training provenance**. The 195 Vim/Git samples were used in previous research, so they are not fresh blind data either. The 15 charset-normalizer cases were selected only when original bytes and their upstream-declared encoding formed valid benchmark cases; they are too few to infer broad accuracy.

The original charset-normalizer project contains **18 labeled `sample-*.txt` fixtures**, with encoding labels listed in its own `tests/test_full_detection.py`. Earlier benchmark versions incorrectly treated all of them as UTF-8. The final benchmark uses those 18 *authoritative* codec labels, validates each source, and then globally deduplicates and selects qualifying records, leaving 15 in the five corpora.

These sets are suitable as a **fixed engineering regression / comparative diagnostic**. They are **not** a measure of accuracy in the wild; do not advertise `95.49%` as CodecCat's general accuracy.

## Reproducibility

- [Workflow](../.github/workflows/five-corpus-benchmark.yml)
- [Benchmark code](../experiments/five_corpus_benchmark.py)
- [Run with saved manifest and per-corpus results](https://github.com/Johnny-Kao/CodecCat/actions/runs/42136798760)
- Pin chardet/test-data: `b0c0d206913ba118ffea2f7d306883e7a4a76db7`
- Pin charset-normalizer fixtures: `7d74fe4a8341ae27a8f42c759793301dbd304e10`
- Pin Vim: `88d96a15f99851faca0cacbbbdcede3f8c25149b`
- Pin Git: `0ffa3e35c7c82ab8d0de226149fb03b3f7674d05`

The runner never uploads raw fixture bytes. The manifest stores per-file content fingerprints, codec counts, source pools and immutable commit IDs. The metrics use exact strictly decoded Unicode text and normalized historical codec labels, with separate init, CPU, wall, and peak-RSS reporting.

**Measurement caveats:** Three libraries perform somewhat different work inside their detection APIs; the benchmark separately performs strict decoding for competitor outputs. Each group's CPU total is from a single timed pass with two untimed warm/repeat passes. Memory reflects the whole Python process. Time results on hosted CI runners are noisy; compare direction, not tiny differences between versions. Encoding labels can be ambiguous even if Unicode output is identical.

## Next quality improvement

The **195 Vim/Git cases** are the most actionable failure cluster: CodecCat correct text 169/195 versus chardet 192/195. Fixes should be driven by *per-encoding failure traces* and verified using new, source-disjoint naturally encoded files **not seen during the development of the changes**. Do not optimize on the benchmark and call the same benchmark blind confirmation.
