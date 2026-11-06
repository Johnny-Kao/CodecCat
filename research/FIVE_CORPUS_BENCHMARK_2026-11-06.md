# CodecCat: five-corpus three-detector benchmark (2026-11-06)

**Status: CI PASS.** [Run #42135919470](https://github.com/Johnny-Kao/CodecCat/actions/runs/42135919470) · [Test script](../experiments/five_corpus_benchmark.py) · [Workflow](../.github/workflows/five-corpus-benchmark.yml)

## Frozen experiment

- **Released artifact:** CodecCat Native `v0.1.0a3` Linux x86-64 Wheel from GitHub Releases, installed with `--no-deps`.
- **Comparators:** `chardet==7.7.0`, `charset-normalizer==3.5.2`.
- **Runtime:** Ubuntu 24.04 / Python 3.12, separately launched processes per engine per corpus, single-threaded BLAS env.
- **Five fixed seeded corpora** (seed `20261106`), 836 samples, deduplicated byte SHA256 and grouped by source document.
- **Pools:** 633 cases from `chardet/test-data` (part of the released model's training source); 8 strictly valid original `charset-normalizer` text fixtures; 195 original Vim/Git translation-file windows, previously examined during CodecCat research.
- Source revisions pinned in the workflow and manifest; raw third-party source data are **not** uploaded to GitHub Actions artifacts.
- The test measures exact decoded Unicode text, exact normalized codec label, call CPU time, wall-time p50/p95, and process-wide peak RSS. API work is not perfectly identical; all comparators are strictly decoded for the text metric.

## Results — all five corpora

| Metric | CodecCat Native | chardet 7.7.0 | charset-normalizer 3.5.2 |
|---|---:|---:|---:|
| Correct Unicode text, of 836 | 799 (95.574%) | **822 (98.325%)** | 691 (82.656%) |
| Exact codec label, of 836 | **764 (91.388%)** | 707 (84.569%) | 575 (68.780%) |
| Total detection+decoding CPU, ms | **650.101** | 784.969 | 891.627 |
| p50 wall latency across corpora, ms | 0.800–0.879 | **0.406–0.658** | 0.631–0.889 |
| Peak process RSS across corpora, MiB | 66.3–68.3 | 60.8–63.9 | **40.0–43.8** |

## Per-corpus correct decoded text

| Corpus | Cases | CodecCat | chardet | charset-normalizer |
|---|---:|---:|---:|---:|
| 1 | 182 | 170 | **179** | 147 |
| 2 | 150 | **149** | 146 | 132 |
| 3 | 163 | 154 | **162** | 142 |
| 4 | 162 | 154 | **158** | 133 |
| 5 | 179 | 172 | **177** | 137 |

## Source-specific audit — critical

| Source pool | Cases | CodecCat | chardet | charset-normalizer |
|---|---:|---:|---:|---:|
| chardet/test-data, overlaps CodecCat training | 633 | 622 (98.26%) | 622 (98.26%) | 508 (80.25%) |
| charset-normalizer fixtures | 8 | 8 (100%) | 8 (100%) | 8 (100%) |
| Vim / Git previously examined documents | 195 | **169 (86.67%)** | 192 (98.46%) | 175 (89.74%) |

**Interpretation:** CodecCat is not best at correctly decoding text on this benchmark; chardet is better, and even charset-normalizer outperforms CodecCat on the 195 Vim/Git cases. The full-corpus label metric favors CodecCat but has limited real-world meaning when multiple codecs yield identical text. CodecCat's CPU sum is lower and p95 is generally lower, while per-call median latency and peak RSS favor competitors. The 8 charset-normalizer samples are too few to support a meaningful separate claim.

## Limits that prevent a general accuracy claim

1. **Training overlap:** The chardet pool contains texts used by CodecCat's default model. None of the five mixed corpora is a clean blind acceptance test. The aggregate 95.574% is **not** a population accuracy estimate.
2. **Previously inspected sources:** Vim/Git data were examined in earlier research; not fresh blind data.
3. **Small competitor fixture pool:** Only 8 strictly UTF-8-decodable original charset-normalizer test files qualified under the conservative source-label rule.
4. **Header and window transformation:** PO charset declaration strings were masked and windows were selected deterministically from existing catalog bytes. Ground truth uses source-declared charset.
5. **Workload and API differences:** Different detectors do different internal work; CPU measurements cover complete detect+strict decoding for all, and may vary between runs. Peak RSS is total process memory, not incremental detector memory.
6. **No cross-source near-duplicate audit:** Source-document grouping and raw SHA256 dedup do not prove text-level disjointness across different upstream repositories.

## Recommended next engineering decision

Prioritize **incorrect text recovery on Vim/Git** and report per-encoding errors with comparable case traces. Before publishing headline accuracy claims, build an untouched, source-disjoint external holdout with verified codec labels and sufficient non-UTF-8 representation. Do not optimize models against the five benchmark sets and then call the same sets blind validation.

**Publishing:** no code/model/Release/PyPI change from this measurement. Keep the existing README's results properly qualified until stronger confirmation.
