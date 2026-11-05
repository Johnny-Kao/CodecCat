# CodecCat

**A Python library that detects text encodings and decodes text.**

Give it bytes, and CodecCat helps you read them—without guessing whether they are UTF-8, Shift-JIS, Big5, or something else.

## Install

Requires Python 3.10+ on Linux, macOS, or Windows.

1. Download the **Native v0.1.0a3** wheel for your operating system and CPU from [GitHub Releases](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.0a3).
2. Install the downloaded `.whl` file:

```bash
python -m pip install ./codeccat-0.1.0a3-*.whl
```

Choose the wheel that matches your system. The native version does not require NumPy or scikit-learn. **Not yet on PyPI**—`pip install codeccat` alone is not supported.

## Python

```python
import codeccat

data = "こんにちは".encode("shift_jis")
result = codeccat.detect(data)

print(result.encoding)  # detected encoding
print(result.text)      # decoded text
```

Only need the text?

```python
text = codeccat.decode(data)  # string, or None if decoding fails
```

## Command line

```bash
codeccat file.txt
codeccat --decode file.txt > readable.txt
```

## Comparison

Tested on the same **304 text samples** using Python 3.12 on Linux:

| | **CodecCat** | **chardet** | **charset-normalizer** |
|---|---:|---:|---:|
| Text decoded correctly | **294 / 304** | 286 / 304 | 249 / 304 |
| Typical detection time | **0.47 ms** | 0.54 ms | 0.89 ms |
| Slower-case detection time (p95) | **1.49 ms** | 3.62 ms | 2.77 ms |
| Peak memory | 55 MB | 52 MB | **26 MB** |
| Installed environment size | 30 MB | 16 MB | **14 MB** |

**What this means:** CodecCat decoded more of these samples correctly and was faster in this test, but used more memory and disk space.

**Important:** These samples include data used during development, so the correct-text counts **are not an independent accuracy rating**. Different libraries also perform somewhat different work when called. Results may vary with your files and machine. [Benchmark details and methodology](docs/BENCHMARKS.md).

## Notes

- Some text encodings cannot be distinguished from the bytes alone. Check `result.ambiguous` if the exact encoding matters.
- CodecCat never silently substitutes replacement characters when decoding.
- This is an **Alpha release**; feedback and bug reports are welcome.

[Releases](https://github.com/Johnny-Kao/CodecCat/releases) · [Report an issue](https://github.com/Johnny-Kao/CodecCat/issues) · [Benchmark details](docs/BENCHMARKS.md) · [MIT License](LICENSE) · [Rights inquiries](RIGHTS_AND_PROVENANCE.md)
