# CodecCat

**Fast character-encoding detection for Python, backed by native inference and strict decoding.**

[![Release](https://img.shields.io/github/v/release/Johnny-Kao/CodecCat)](https://github.com/Johnny-Kao/CodecCat/releases)
[![Python](https://img.shields.io/badge/Python-3.10--3.14-blue)](https://github.com/Johnny-Kao/CodecCat)
[![License](https://img.shields.io/github/license/Johnny-Kao/CodecCat)](LICENSE)

```python
import codeccat

data = "こんにちは".encode("shift_jis")

result = codeccat.detect(data)

print(result.encoding)
print(result.text)
```

CodecCat detects the character encoding of byte input and decodes it to Unicode.

It is designed for applications that need predictable decoding behavior, low detection latency, and explicit handling of ambiguous input.

## Why CodecCat?

- **Strict decoding** — invalid text is not silently replaced.
- **Native inference** — the default detector does not require NumPy or scikit-learn.
- **Explicit ambiguity** — callers can inspect `result.ambiguous` instead of assuming every detection is certain.
- **Portable wheels** — published for Linux, macOS, and Windows on supported x86-64 and ARM64 targets.
- **Stable public API** — the 0.1.x line has a documented compatibility contract.

Encoding detection is inherently probabilistic: some byte sequences can validly represent more than one encoding. CodecCat exposes that uncertainty rather than treating every result as uniquely identifiable.

## Install

CodecCat v0.1.0 is currently distributed through GitHub Releases.

Download the wheel for your operating system and Python version from:

[CodecCat v0.1.0 — Stable](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.0)

Then install it:

```bash
python -m pip install ./codeccat-0.1.0-*.whl
```

CodecCat supports Python 3.10–3.14 on the published Linux (`manylinux_2_28+`), macOS, and Windows targets.

On Windows, use the exact downloaded wheel filename instead of the `*` wildcard.

PyPI publishing is a separate distribution step and is not currently claimed here.

## Usage

### Detect and decode

```python
import codeccat

data = "こんにちは".encode("shift_jis")
result = codeccat.detect(data)

print(result.encoding)
print(result.text)
print(result.ambiguous)
```

### Decode directly

If you only need the decoded text:

```python
text = codeccat.decode(data)
```

### Command line

Inspect a file:

```bash
codeccat file.txt
```

Decode it:

```bash
codeccat --decode file.txt > readable.txt
```

## Performance

CodecCat v0.1.0 was compared with `chardet` and `charset-normalizer` using the same 843 text samples across five fixed test sets on Ubuntu 24.04 / Python 3.12.

| Metric | CodecCat | chardet | charset-normalizer |
| --- | ---: | ---: | ---: |
| Correctly decoded text | 97.98% | **98.34%** | 82.80% |
| Exact encoding name | **93.36%** | 84.46% | 69.16% |
| Total CPU time | **779 ms** | 1,008 ms | 1,205 ms |
| Typical latency (p50 range) | 0.94–1.04 ms | **0.49–0.83 ms** | 0.85–1.16 ms |
| Slower-case latency (p95 range) | **1.38–1.65 ms** | 3.23–3.86 ms | 3.55–5.15 ms |
| Peak RAM | 66–69 MiB | 61–64 MiB | **40–44 MiB** |

CodecCat is not uniformly better on every metric.

In this benchmark, `chardet` correctly decoded 3 more samples, while CodecCat identified the exact encoding more often, used less total measured CPU time, and showed substantially lower p95 latency. CodecCat used more peak memory.

These measurements were collected on a shared CI runner. Small timing differences should not be interpreted as statistical guarantees.

### Accuracy limitation

The 843-sample benchmark includes training and previously examined source texts. The reported 97.98% result is therefore an engineering benchmark, **not an independently estimated real-world accuracy rate**.

A source-level paired GNOME Software test showed 18 additional correct decodes with the same frozen model without observed regressions in that test set.

See [benchmark methodology and limitations](docs/BENCHMARKS.md) for the reproducible test setup and interpretation.

## Detection semantics

Character encoding is not always uniquely identifiable from bytes alone.

For applications where the exact encoding matters:

```python
result = codeccat.detect(data)

if result.ambiguous:
    # Apply application-specific handling.
    ...
```

CodecCat uses strict decoding behavior. Invalid byte sequences are not silently converted using replacement characters.

Encoding inference remains probabilistic and cannot guarantee a uniquely correct answer for arbitrary input.

## Stability

CodecCat v0.1.0 is the first stable release.

The `0.1.x` line preserves the documented public API, including the primary detection and decoding calls and documented result fields.

See the [public API compatibility contract](docs/STABILITY.md).

## Distribution and provenance

Published wheels are built for supported Linux, macOS, and Windows targets. The frozen default model is distributed with the package and verified as part of the release process.

Model and source provenance, including third-party rights considerations, are documented separately.

See [Rights and Provenance](RIGHTS_AND_PROVENANCE.md).

## Links

[Releases](https://github.com/Johnny-Kao/CodecCat/releases) ·
[Issues](https://github.com/Johnny-Kao/CodecCat/issues) ·
[Benchmarks](docs/BENCHMARKS.md) ·
[Stability](docs/STABILITY.md) ·
[License](LICENSE)
