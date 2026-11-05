# CodecCat Native v0.1.0a3 — Cross-platform / resource benchmark (2026-11-05)

**Status:** staging research only, branch `refactor/native-inference-v1`; no release or merge.
**Evidence:** [6-platform CI #42040340016](https://github.com/Johnny-Kao/CodecCat/actions/runs/42040340016) (PASS);
[isolated 4-engine resource CI #42040736961](https://github.com/Johnny-Kao/CodecCat/actions/runs/42040736961) (PASS).

## Cross-platform

| Host | Runner | Build | Native public API parity | Native L2 numerical parity |
|---|---|---|---|---|
| Linux x86-64 | ubuntu-24.04 | PASS | PASS | PASS |
| Linux AArch64 | ubuntu-24.04-arm | PASS | PASS | PASS |
| macOS x86-64 | macos-15-intel | PASS | PASS | PASS |
| macOS AArch64 | macos-15 | PASS | PASS | PASS |
| Windows x86-64 | windows-2025 | PASS | PASS | PASS |
| Windows ARM64 | windows-11-arm | PASS | PASS | PASS |

All use identical CCM1 serialized pretrained Alpha stable23 parameters, generated
once by the separate reference job; 6720 exact API result comparisons including
optional L3 and BOM policies plus 267 numerical L2 comparisons. In each clean
interpreter, NumPy, SciPy, scikit-learn and joblib were absent after wheel install.
This is Python 3.12 on each runner, not a full Python version matrix. Platform
throughput/RSS were **not** measured in the matrix.

## Real-resource comparison (same Linux x86-64 CI runner)

304 identical inputs drawn from the *previously used chardet development corpus*
(80 naturally short rows, 220 source-distributed long rows, 4 edge inputs).
One subprocess and fresh environment per library; no retraining on the measured
items. It is **NOT** independent confirmation of any detector's accuracy. Native
and Alpha share the shipped `stable23` classifier; the previously reported
94.243% five-panel research classifier is a different model.

| Metric | Native | Alpha | chardet 7.5.0 | charset-normalizer 3.5.2 |
|---|---:|---:|---:|---:|
| Correct Unicode decoded text (out of 304) | 295 | 295 | 286 | 249 |
| p50 wall/call ms | 0.3111 | 0.7913 | 0.2972 | 0.4571 |
| p95 wall/call ms | 1.0592 | 2.6069 | 1.9600 | 1.4825 |
| p50 process CPU/call ms | 0.3097 | 0.7892 | 0.2960 | 0.4559 |
| Sum process CPU ms | 126.27 | 332.62 | 179.74 | 182.24 |
| Peak process RSS MiB | 87.20 | 213.21 | 51.59 | 25.95 |
| Isolated venv disk MB (du -sm) | 45 | 278 | 16 | 14 |

Interpretation:
- Native uses **~62.0% less total process CPU** than old Alpha on these cases;
  Alpha/Native have exactly the same 295 correct cases.
- Native peak RSS **~59.1% less** than Alpha (87.2 vs 213.21 MiB),
  but **~69% higher** than chardet and **~3.36x** charset-normalizer.
- Native environment occupies 45 MB, **larger** than chardet's 16 MB and
  charset-normalizer's 14 MB; this is largely dominated by the bundled model.
- Native p50 is **slightly slower** than chardet (0.3111 vs 0.2972 ms),
  despite better p95/aggregate CPU. Do not claim speed superiority on all sizes.
- Source Alpha `detect()` both selects a codec and strict-decodes text;
  chardet `detect()` selects encoding and does not decode, and
  charset-normalizer `from_bytes().best()` uses different internal work.
  We only applied common strict decoding **after timing** for comparison of
  correctness. Therefore measured costs are costs of each library's public
  detection API, not identical-stage kernel timing.
- The 97.04% (295/304) subset score **must not** be represented as an overall
  97% accuracy claim or substituted for the pre-existing 94.24% *research*
  result. This sample is highly source-specific and already used in development.
- CPU, RSS and installation sizes are for one Linux CI runner and specific
  versions/builds, not a universal ranking.

## Native package constraints before release

1. Reduce the 32,479,080-byte uncompressed CCM1 model footprint **without**
   changing decoded Unicode output, model ranking or introducing inference
   heavyweight dependencies. Compare compressed read-only artifact lazy-load,
   sparse weight representation, and only later precision quantization.
2. Profile cold import vs loaded model, independently. Import may be dominated
   by the Python adapter; model load already improved versus Alpha.
3. Check packaging and Python 3.10 through current versions, wheel audit,
   universal2/abi3 portability, malformed model fuzzing and thread safety.
4. Establish explicit Legacy R21 API migration/deprecation strategy.
5. Preserve the 6-platform parity gates and separate Python training.
6. Research model (binary12) still requires independent export and validation;
   successful Alpha model parity does not prove the 94.24% research baseline.

**Decision:** Cross-platform feature parity passes. Do not merge to main or
publish new release yet; prioritize model memory footprint and release hardening.
