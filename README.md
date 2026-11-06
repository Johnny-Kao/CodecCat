<div align="center">

# CodecCat

**Detect the encoding. Read the text.**

A Python library for opening text when you don't know its character encoding.

[![Release](https://img.shields.io/github/v/release/Johnny-Kao/CodecCat?include_prereleases&label=version)](https://github.com/Johnny-Kao/CodecCat/releases)
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Native tests](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-platform-matrix.yml/badge.svg?branch=main)](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-platform-matrix.yml)

</div>

## Install

Download the wheel for your operating system from [CodecCat v0.1.0a4](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.0a4), then install it:

```bash
python -m pip install ./codeccat-0.1.0a4-*.whl
```

Supports **Python 3.10+** on Linux, macOS and Windows. On Windows, use the exact downloaded wheel filename instead of the `*` wildcard. Native detection doesn't require NumPy or scikit-learn. **Not on PyPI yet.**

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

Tested with the released **v0.1.0a4 Native Wheel** against chardet and charset-normalizer on the **same 843 samples** across five fixed test sets.

| | CodecCat | chardet | charset-normalizer |
|---|---:|---:|---:|
| Correctly decoded text | 97.15% | **98.34%** | 82.80% |
| Exact encoding name | **92.53%** | 84.46% | 69.16% |
| Total CPU time | **601 ms** | 801 ms | 1,061 ms |
| Typical latency (p50 range) | 0.75–0.79 ms | **0.43–0.74 ms** | 0.72–1.00 ms |
| Slower-case latency (p95 range) | **1.11–1.18 ms** | 2.21–2.81 ms | 3.04–4.67 ms |
| Peak RAM (range) | 66–69 MiB | 61–64 MiB | **40–44 MiB** |

**chardet decoded more text correctly.** CodecCat had lower CPU time and p95 latency in this run, but used more RAM. On 285 separately sourced GLib text samples, v0.1.0a4 decoded 44 more correctly than v0.1.0a3, with zero observed regressions.

**About the test:** The 843 samples include training and development data. Results describe these test sets, **not general real-world accuracy**; latency and CPU measurements are from one CI run. [Full methodology and raw test links](docs/BENCHMARKS.md#five-corpus-benchmark-released-native-v010a4).


## Notes

Encoding isn't always identifiable from bytes alone. Check `result.ambiguous` when the exact encoding matters. Decoding is strict: invalid text is not silently replaced. CodecCat v0.1.0a4 is an **Alpha release**.

[Releases](https://github.com/Johnny-Kao/CodecCat/releases) · [Issues](https://github.com/Johnny-Kao/CodecCat/issues) · [Benchmarks](docs/BENCHMARKS.md) · [MIT License](LICENSE) · [Rights inquiries](RIGHTS_AND_PROVENANCE.md)
