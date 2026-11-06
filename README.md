<div align="center">

# CodecCat

**Detect the encoding. Read the text.**

A Python library for opening text when you don't know its character encoding.

[![Release](https://img.shields.io/github/v/release/Johnny-Kao/CodecCat?include_prereleases&label=version)](https://github.com/Johnny-Kao/CodecCat/releases)
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Native tests](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-platform-matrix.yml/badge.svg?branch=main)](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-platform-matrix.yml)

</div>

## Install

Download the wheel for your operating system from [CodecCat v0.1.0](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.0), then install it:

```bash
python -m pip install ./codeccat-0.1.0-*.whl
```

Supports **Python 3.10+** on Linux, macOS and Windows. On Windows, use the exact downloaded wheel filename instead of the `*` wildcard. Native detection doesn't require NumPy or scikit-learn. **GitHub Wheels; PyPI publishing is a separate distribution step.**

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

## Comparison

CodecCat **v0.1.0 stable candidate**, using the same frozen model and decoder code as the verified v0.1.0a5 baseline, compared with chardet and charset-normalizer on the **same 843 text samples** across five fixed test sets (Ubuntu 24.04 / Python 3.12).

| | CodecCat | chardet | charset-normalizer |
|---|---:|---:|---:|
| Correctly decoded text | 97.98% | **98.34%** | 82.80% |
| Exact encoding name | **93.36%** | 84.46% | 69.16% |
| Total CPU time | **660 ms** | 861 ms | 1,073 ms |
| Typical latency (p50 range) | 0.78–0.89 ms | **0.46–0.75 ms** | 0.74–1.02 ms |
| Slower-case latency (p95 range) | **1.20–1.32 ms** | 2.69–3.01 ms | 3.10–4.64 ms |
| Peak RAM (range) | 66–69 MiB | 61–64 MiB | **40–44 MiB** |

**chardet still decoded slightly more of these samples correctly.** CodecCat used less measured CPU time and had lower p95 latency, but used more RAM.

**Limits:** These test sets include model training and development sources; **97.98% is not a general real-world accuracy estimate**. CPU and timing values come from one hosted CI run. A separately pinned GNOME Software source showed **198/225 → 216/225** correct decodes versus v0.1.0a4, using the identical model and with no observed regressions. [Full methods and caveats](docs/BENCHMARKS.md#five-corpus-benchmark-released-native-v010a5).

## Notes

[Public API compatibility contract](docs/STABILITY.md): the 0.1.x stable line preserves the documented public calls and result fields. Encoding inference remains probabilistic and is not guaranteed for arbitrary bytes.


Encoding isn't always identifiable from bytes alone. Check `result.ambiguous` when the exact encoding matters. Decoding is strict: invalid text is not silently replaced. CodecCat v0.1.0 is a **stable release**.

[Releases](https://github.com/Johnny-Kao/CodecCat/releases) · [Issues](https://github.com/Johnny-Kao/CodecCat/issues) · [Benchmarks](docs/BENCHMARKS.md) · [MIT License](LICENSE) · [Rights inquiries](RIGHTS_AND_PROVENANCE.md)
