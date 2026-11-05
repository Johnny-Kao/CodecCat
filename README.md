# CodecCat

**Don't know the encoding of a text file? CodecCat detects it and helps you read it.**

Give it raw bytes. Get the encoding and decoded text back. No need to guess between UTF-8, Shift-JIS, Big5, or other character encodings.

## Try it

```python
import codeccat

data = "こんにちは".encode("shift_jis")
result = codeccat.detect(data)

print(result.encoding)  # shift_jis
print(result.text)      # こんにちは
```

Just want the text?

```python
text = codeccat.decode(data)  # Returns a string, or None if decoding fails
```

You can also use it from the command line:

```bash
codeccat file.txt
codeccat --decode file.txt > readable.txt
```

## Install

CodecCat is currently available as an **experimental GitHub release**. Install the latest published version (v0.1.0a2) with Python 3.10+:

```bash
python -m pip install "https://github.com/Johnny-Kao/CodecCat/releases/download/v0.1.0a2/codeccat-0.1.0a2-py3-none-any.whl"
```

**Not on PyPI yet** — `pip install codeccat` is not available. A faster Rust-powered version is being prepared but has **not been released**.

## How fast is it?

In a Linux benchmark of 304 text samples, the upcoming native version took:

| Library | Typical detection time (median) |
|---|---:|
| **CodecCat Native** | **0.47 ms** |
| chardet | 0.54 ms |
| charset-normalizer | 0.89 ms |

These are measurements from one development benchmark, **not guaranteed speeds**. The libraries do somewhat different work, and CodecCat uses more memory than these competitors. [See the full benchmark and test method](docs/BENCHMARKS.md).

## Good to know

- Some encodings are impossible to distinguish from bytes alone. Check `result.ambiguous` when it matters.
- CodecCat won't silently replace broken text with incorrect characters.
- The native version is tested on Windows, macOS, and Linux, but isn't the currently downloadable version.

## Links

[Releases](https://github.com/Johnny-Kao/CodecCat/releases) · [Report a problem](https://github.com/Johnny-Kao/CodecCat/issues) · [Benchmarks](docs/BENCHMARKS.md) · [MIT License](LICENSE) · [Data sources and rights inquiries](RIGHTS_AND_PROVENANCE.md)
