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

Supports **Python 3.10–3.14** on the published Linux (manylinux 2.28+), macOS and Windows targets. On Windows, use the exact downloaded wheel filename instead of the `*` wildcard. Native detection doesn't require NumPy or scikit-learn. **GitHub Wheels; PyPI publishing is a separate distribution step.**

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

Tested with the **published Stable v0.1.0 manylinux Wheel**, chardet and charset-normalizer on exactly the same **843 text samples** across five fixed sets (Ubuntu 24.04 / Python 3.12).

| | CodecCat | chardet | charset-normalizer |
|---|---:|---:|---:|
| Correctly decoded text | 97.98% | **98.34%** | 82.80% |
| Exact encoding name | **93.36%** | 84.46% | 69.16% |
| Total CPU time | **779 ms** | 1,008 ms | 1,205 ms |
| Typical latency (p50 range) | 0.94–1.04 ms | **0.49–0.83 ms** | 0.85–1.16 ms |
| Slower-case latency (p95 range) | **1.38–1.65 ms** | 3.23–3.86 ms | 3.55–5.15 ms |
| Peak RAM (range) | 66–69 MiB | 61–64 MiB | **40–44 MiB** |

**chardet decoded slightly more text correctly.** CodecCat used less measured CPU and lower p95 latency, but more RAM. Results are from one shared CI runner; small speed differences are not statistical guarantees.

**Accuracy limitation:** This benchmark includes training and previously examined source texts; **97.98% is not an independently estimated real-world accuracy**. A source-level paired GNOME Software test showed 18 correct decodes gained with the same frozen model, without observed regressions. [Official Stable CI and methodology](docs/BENCHMARKS.md#stable-v010-published-wheel-2026-11-06).

## Notes

[Public API compatibility contract](docs/STABILITY.md): the 0.1.x stable line preserves the documented public calls and result fields. Encoding inference remains probabilistic and is not guaranteed for arbitrary bytes.

Encoding isn't always identifiable from bytes alone. Check `result.ambiguous` when the exact encoding matters. Decoding is strict: invalid text is not silently replaced. CodecCat v0.1.0 is a **stable release**.

[Releases](https://github.com/Johnny-Kao/CodecCat/releases) · [Issues](https://github.com/Johnny-Kao/CodecCat/issues) · [Benchmarks](docs/BENCHMARKS.md) · [MIT License](LICENSE) · [Rights inquiries](RIGHTS_AND_PROVENANCE.md)
