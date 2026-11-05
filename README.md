<div align="center">

# CodecCat

**Detect the encoding. Read the text.**

A Python library for opening text when you don't know its character encoding.

[![Release](https://img.shields.io/github/v/release/Johnny-Kao/CodecCat?include_prereleases&label=version)](https://github.com/Johnny-Kao/CodecCat/releases)
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Native tests](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-platform-matrix.yml/badge.svg?branch=main)](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-platform-matrix.yml)

</div>

## Install

Download the wheel for your operating system from [CodecCat v0.1.0a3](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.0a3), then install it:

```bash
python -m pip install ./codeccat-0.1.0a3-*.whl
```

Supports **Python 3.10+** on Linux, macOS and Windows. Native detection doesn't require NumPy or scikit-learn. **Not on PyPI yet.**

## Use it

```python
import codeccat

data = "こんにちは".encode("shift_jis")
result = codeccat.detect(data)

print(result.encoding)  # detected encoding
print(result.text)      # decoded text
```

Only need the text? Use `codeccat.decode(data)`.

From the command line:

```bash
codeccat file.txt
codeccat --decode file.txt > readable.txt
```

## Accuracy

Previous **stable23 model** evaluations used text from sources outside its training corpus. The numbers below count **correctly decoded text**, not necessarily an exact encoding-name match.

| Test source | CodecCat | chardet | charset-normalizer |
|---|---:|---:|---:|
| Vim translation files (159 samples) | 155/159 | **159/159** | 143/159 |
| Git translation files (76 samples) | 76/76 | 76/76 | 76/76 |
| Tatoeba CC0 text, re-encoded (560 samples) | 524/560 | **554/560** | 498/560 |

These are **historical stable23 tests**, not a fresh independent evaluation of the downloaded Native wheel. The Tatoeba test uses synthetically re-encoded text; the Vim and Git samples contain related windows from the same files. Results vary by source, and chardet performed better on two of these tests.

[Accuracy test details and original CI evidence](docs/BENCHMARKS.md#accuracy-across-data-sources)

## Speed and resources

A separate Linux/Python 3.12 benchmark measured 304 development samples:

| | CodecCat Native | chardet | charset-normalizer |
|---|---:|---:|---:|
| Median detection time | **0.47 ms** | 0.54 ms | 0.89 ms |
| Peak process memory | 55 MiB | 52 MiB | **26 MiB** |
| Installed environment size | 30 MB | 16 MB | **14 MB** |

CodecCat was faster on this workload, but used more memory and disk space. Different libraries also perform different work during their API calls. **The 304 development samples are not an independent accuracy benchmark.**

[Full performance results and methodology](docs/BENCHMARKS.md#performance-and-resources)

## Notes

Encoding isn't always identifiable from bytes alone. Check `result.ambiguous` when the exact encoding matters. Decoding is strict: invalid text is not silently replaced. CodecCat v0.1.0a3 is an **Alpha release**.

[Releases](https://github.com/Johnny-Kao/CodecCat/releases) · [Issues](https://github.com/Johnny-Kao/CodecCat/issues) · [Benchmarks](docs/BENCHMARKS.md) · [MIT License](LICENSE) · [Rights inquiries](RIGHTS_AND_PROVENANCE.md)
