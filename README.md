# CodecCat

[![Native platform CI](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-platform-matrix.yml/badge.svg?branch=main)](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-platform-matrix.yml)
[![Python compatibility](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-python-compat.yml/badge.svg?branch=main)](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-python-compat.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**CodecCat is a Python package for character-encoding detection and strict Unicode decoding.** Its native implementation uses Rust for trained-model inference and Python's standard codecs for decoding. It returns the detected encoding, decoded text, and ambiguity information through one API.

**Project links:** [Source](https://github.com/Johnny-Kao/CodecCat) · [Releases](https://github.com/Johnny-Kao/CodecCat/releases) · [Issues](https://github.com/Johnny-Kao/CodecCat/issues) · [Benchmarks](docs/BENCHMARKS.md) · [License](LICENSE)

> **Release status:** The latest published GitHub Alpha is **[v0.1.0a2](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.0a2)** (Python-based). The Rust-native `0.1.0a3` implementation is on `main` but **has not been released**. CodecCat is **not on PyPI**, so `pip install codeccat` is not available yet.

## Why CodecCat?

- **Detection and decoding together:** get an encoding label and strictly decoded text, rather than only a label.
- **Explicit uncertainty:** inspect `ambiguous`, `candidates`, and `reason` instead of assuming every byte sequence has a unique encoding.
- **Native inference:** Rust/PyO3 L2 scoring with a compact bundled CCM2 float32 model.
- **Lightweight runtime dependencies:** the native detector needs no NumPy, SciPy, or scikit-learn; these are separate from offline training.
- **Portable wheels:** Linux, macOS, and Windows on x86-64 and ARM64 have passed installed-wheel CI tests.

## Quick start

```python
import codeccat

raw = "日本語と café".encode("utf-8")
result = codeccat.detect(raw)

print(result.encoding)   # selected codec, or None
print(result.text)       # strictly decoded text, or None
print(result.ambiguous)  # True when the label is not uniquely identifiable

text = codeccat.decode(raw)  # str | None
```

For a custom model or optional language evidence:

```python
from codeccat import CodecCat

detector = CodecCat.from_files("my-model.ccm")
result = detector.detect(b"sample text")

with_language_evidence = CodecCat(use_l3=True)  # L3 is off by default
```

Detection accepts a complete byte sequence, **not a streaming fragment**. A successful decode does not prove the selected encoding label is uniquely correct.

## Installation

The currently **published v0.1.0a2 Alpha** can be installed directly from its GitHub Release:

```bash
python -m pip install "https://github.com/Johnny-Kao/CodecCat/releases/download/v0.1.0a2/codeccat-0.1.0a2-py3-none-any.whl"
```

The newer Rust-native source on `main` requires **Python 3.10+** and a platform-specific wheel. Its default trained model is generated during the official build and is **not committed to the source tree**; a bare `pip install .` from an unprepared checkout does not provide a usable default model. See [native build validation](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-root-package.yml). No PyPI publishing is configured.

## Command line

```bash
codeccat document.txt
codeccat --json document.txt
codeccat --decode document.txt > decoded.txt
cat document.txt | python -m codeccat --json
```

The CLI accepts up to **16 MiB** per input and fails instead of silently truncating. Decoding uses strict error handling; a recognized BOM is stripped by default (use `--bom-policy preserve` to retain it).

## Benchmarks

Measured on a **Linux x86-64 GitHub Actions runner** (2026-11-05), using the optimized Native CCM2 implementation:

| Metric | CodecCat Native | Previous CodecCat Alpha |
|:--|--:|--:|
| 4 KB detection, p50 | **0.1908 ms** | 2.4429 ms |
| Model load | **41.3 ms** | 838.5 ms |
| Peak process RSS | **54.7 MiB** | 212.7 MiB |
| Uncompressed model file | **16.64 MB** | 32.48 MB |

On a separate **304-input mixed-size development benchmark**, CodecCat Native p50 was **0.4712 ms**, compared with **0.5388 ms** for chardet 7.5.0 and **0.8903 ms** for charset-normalizer 3.5.2. The benchmark also showed higher peak memory than both competitors; relative performance varies by input size and API semantics.

**Benchmark scope matters:** these are particular CI-runner measurements, not platform-wide speed guarantees. The 304 inputs include previously used development data and **must not** be interpreted as an independent accuracy test. CodecCat performs strict decoding inside its timed detection API; the competing APIs do different work.

[Full results, versions, methodology, limitations, and reproducibility](docs/BENCHMARKS.md) · [Underlying optimization report](native-prototype/P0_P2_OPTIMIZATION_RESULTS_2026-11-05.md)

## How it works

```text
Input bytes
    │
    ├─ L1: structural checks and BOM handling
    │
    ├─ L2: native Rust scoring of byte n-gram features
    │       └─ bundled CCM2 model (float32 weights)
    │
    ├─ L3: optional decoded-text coherence reranking (off by default)
    │
    └─ Python strict codec decoding → detection result
```

The runtime supports CCM1 float64 model loading for compatibility and lazily exposes older Python APIs via the optional `codeccat[legacy]` extra. The independent [`codeccat-train`](native-prototype/training/README.md) tool handles offline training and is **not installed with the detector**.

## Limitations and verification

- Some encodings are indistinguishable for particular inputs; inspect the full result and consider external metadata when available.
- Strict decoding can reject a candidate; it never silently substitutes replacement characters.
- The default model uses the historical Alpha **stable23** parameters, not the separate 94.243%-accuracy research classifier.
- Tests include 6,720 API comparisons against Alpha, malformed-model rejection, concurrency tests, Python 3.10–3.14, and clean-installed native wheels across six operating-system/CPU combinations. These establish compatibility on tested inputs, **not universal accuracy**.

[Six-platform CI](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-platform-matrix.yml) · [Python compatibility CI](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-python-compat.yml) · [Release checklist](RELEASE_CHECKLIST_NATIVE.md)

## Contributing and support

Issues and small, focused pull requests are welcome. Please [open an issue](https://github.com/Johnny-Kao/CodecCat/issues/new) first for proposed API changes, new model families, or significant architectural work. Include a minimal reproducible example and relevant Python/platform details for bug reports.

CodecCat source code is released under the [MIT License](LICENSE). The [model provenance and rights inquiry policy](RIGHTS_AND_PROVENANCE.md) explains the external training sources and provides a contact channel for rights concerns. A model checksum verifies file integrity; it does not establish publisher authenticity.
