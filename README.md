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

## Speed

A Linux test with 304 text samples measured median detection time:

| Library | Time |
|---|---:|
| **CodecCat Native** | **0.47 ms** |
| chardet | 0.54 ms |
| charset-normalizer | 0.89 ms |

Results depend on the input and machine. The libraries do different work, and CodecCat uses more memory than the alternatives in this test. [Full benchmark and test details](docs/BENCHMARKS.md).

## Notes

- Some text encodings cannot be distinguished from the bytes alone. Check `result.ambiguous` if the exact encoding matters.
- CodecCat never silently substitutes replacement characters when decoding.
- This is an **Alpha release**; feedback and bug reports are welcome.

[Releases](https://github.com/Johnny-Kao/CodecCat/releases) · [Report an issue](https://github.com/Johnny-Kao/CodecCat/issues) · [Benchmark details](docs/BENCHMARKS.md) · [MIT License](LICENSE) · [Rights inquiries](RIGHTS_AND_PROVENANCE.md)
