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

## Comparison

Tested with the released **v0.1.0a3 Native Wheel** against chardet and charset-normalizer on the **same 843 samples** across five fixed test sets.

| | CodecCat | chardet | charset-normalizer |
|---|---:|---:|---:|
| Correctly decoded text | 95.49% | **98.34%** | 82.80% |
| Exact encoding name | **90.87%** | 84.46% | 69.16% |
| Total CPU time | **598 ms** | 732 ms | 881 ms |
| Typical detection time (p50 range) | 0.73–0.80 ms | **0.36–0.61 ms** | 0.63–0.84 ms |
| Peak RAM (range) | 66–69 MiB | 61–64 MiB | **40–44 MiB** |

**chardet decoded more text correctly.** CodecCat used less CPU overall and had a better slower-case (p95) latency, but used more RAM.

**About the test:** The data includes CodecCat training sources and previously examined files. These figures describe this **specific benchmark**, not independent real-world accuracy. [See the five test sets, sources and full results](docs/BENCHMARKS.md#five-corpus-benchmark-released-native-v010a3).


## Notes

Encoding isn't always identifiable from bytes alone. Check `result.ambiguous` when the exact encoding matters. Decoding is strict: invalid text is not silently replaced. CodecCat v0.1.0a3 is an **Alpha release**.

[Releases](https://github.com/Johnny-Kao/CodecCat/releases) · [Issues](https://github.com/Johnny-Kao/CodecCat/issues) · [Benchmarks](docs/BENCHMARKS.md) · [MIT License](LICENSE) · [Rights inquiries](RIGHTS_AND_PROVENANCE.md)
