# CodecCat

**Fast character-encoding detection for Python, backed by native inference and strict decoding.**

[![Release](https://img.shields.io/github/v/release/Johnny-Kao/CodecCat)](https://github.com/Johnny-Kao/CodecCat/releases)
[![PyPI](https://img.shields.io/pypi/v/codeccat)](https://pypi.org/project/codeccat/)
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

Install the latest Stable release from [PyPI](https://pypi.org/project/codeccat/):

```bash
python -m pip install codeccat
```

To pin the current Stable release:

```bash
python -m pip install codeccat==0.1.1
```

CodecCat ships binary Wheels for Python **3.10–3.14** on supported Linux (glibc 2.28+, x86-64/ARM64), macOS (Intel/Apple Silicon) and Windows (x86-64/ARM64). The current **0.1.1 Stable release includes an sdist**, so systems outside the supported Wheel matrix can attempt a source build if they have a compatible Rust/PyO3 toolchain; this does not guarantee support for every architecture. The initial 0.1.0 release had six Wheels only and remains immutable. The source-distribution round-trip CI validates Python 3.10–3.14.

For offline installs and artifact SHA256 checks, use [GitHub Stable v0.1.1](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.1).


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

### Independent-source confirmation (published Stable 0.1.0)

An additional [pinned external-source benchmark](https://github.com/Johnny-Kao/CodecCat/actions/runs/42261649964) tested the **published Stable Wheel**, not a development build. It screened every eligible byte input against the historical training corpus before scoring (no raw-byte/prefix/sampled-block overlap indicators triggered). Ground truth was strict Unicode text, and all three detectors received the same input bytes.

| External source | CodecCat | chardet 7.7.0 | charset-normalizer 3.5.2 |
| --- | ---: | ---: | ---: |
| GNOME Nautilus UTF-8 PO (124 original files) | 122/124 (98.39%) | **124/124 (100%)** | **124/124 (100%)** |
| CPython CJK fixture pairs (30 original files) | 26/30 (86.67%) | **29/30 (96.67%)** | 25/30 (83.33%) |

Nautilus is a **new external source** for this evaluation. CPython is a previously examined, preregistered confirmation source, not a newly blind sample. The Nautilus group is UTF-8-only, and the CPython set is small and CJK-focused. Neither source represents the real-world mix of encodings; **do not pool the groups into a claimed production accuracy rate**. A negative raw-byte overlap screen is not proof of semantic independence.

### Historical development benchmark limitation

The 843-sample benchmark above includes **633 samples from the model's training source**, plus previously examined development texts. Its 97.98% figure is therefore an **engineering regression benchmark**, not an independent estimate of real-world accuracy. The external-source findings also show that CodecCat can underperform chardet on unseen source families.

A source-level paired GNOME Software test showed 18 additional correct decodes with the same frozen model without observed regressions in that test set.

See [full benchmark methodology and limitations](docs/BENCHMARKS.md) and the [frozen external-source protocol](research/SOURCE_DISJOINT_BENCHMARK_PROTOCOL_2026-11-07.md).

## Optional ambiguity diagnostic (v0.1.1 Stable)

The **published [v0.1.1 Stable](https://pypi.org/project/codeccat/0.1.1/)** includes the **opt-in**, non-mutating C1-control ambiguity diagnostic from [PR #47](https://github.com/Johnny-Kao/CodecCat/pull/47). It is not available in the older `codeccat==0.1.0` Wheel. This diagnostic does not retrain the model or alter default decoding.

With `codeccat==0.1.1`:

```python
from codeccat import detect
from codeccat.diagnostics import inspect_ambiguity

raw = b"Price: \x80; total."
result = detect(raw)
diagnostic = inspect_ambiguity(raw, result)

if diagnostic.warning:
    print(diagnostic.reason, diagnostic.alternative_encodings)
    # Check a trusted source encoding before choosing another interpretation.
```

The function inspects only **already returned model candidates** when selected text contains C1 controls (U+0080–U+009F) and an alternative strictly decodes those bytes without C1 controls. It **does not** change `result.encoding`, `result.text`, `result.candidates`, `result.ambiguous`, BOM handling, or L3 behavior. **A warning does not prove the current decoding is wrong.** Some legal ISO-8859-1 text intentionally contains these controls, and lack of a warning does not prove correctness.

A fixed historical HTML study screened 2,675 files and evaluated 1,886 eligible files after deduplication, strict label decoding, and historical training-source overlap checks. The warning flagged **17 of 572 label-disagreeing results** and none of 1,314 label-agreeing results. Archive labels are not independently verified ground truth, so these numbers are **not** a real-world sensitivity or false-positive rate. See [the validation record](research/C1_STAGING_VALIDATION_2026-11-07.md) and [protocol](research/C1_REAL_HTML_PROTOCOL_2026-11-07.md).

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

## Train your own model (independent PyPI package)

**`codeccat-train 0.1.0` is independently published on [PyPI](https://pypi.org/project/codeccat-train/0.1.0/)** (Trainer release [CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42367573901)). It is not included in the default `pip install codeccat` and is not required for production inference. The runtime stays dependency-light; training uses NumPy and scikit-learn in its own environment.

From a repository checkout, with your own labeled byte files and a [grouped JSONL manifest](native-prototype/training/docs/GETTING_STARTED.md):

```bash
python -m venv .trainer-env
.trainer-env/bin/python -m pip install codeccat-train==0.1.0
.trainer-env/bin/python -m codeccat_train fit \\
  --manifest my-data/manifest.jsonl \\
  --output custom.ccm \\
  --report training-report.json
```

Then load the model using an independent environment containing the **published** Runtime:

```python
from codeccat import CodecCat
custom_detector = CodecCat.from_files("custom.ccm")
result = custom_detector.detect("café".encode("utf-8"))
print(result.encoding, result.text)
```

The Trainer also supports `--precision f32` (compact CCM2), `evaluate` (candidate-vs-unchanged Stable with paired improvements/regressions on labeled bytes), and optional `fit-l3` (separately trained language artifact, disabled by default). The common training engine can be used with **your private dataset** or a separately approved official dataset; supplying your data does not grant the project rights to it.

**Verified:** separate training/Runtime environments, deterministic training, safe manifest checks, native custom-model loading, Linux x64/ARM64 and macOS/Windows checks, and exact historical official L2/L3 artifact reconstruction under pinned sources and dependencies. **Not established:** universal legal clearance for historical datasets, or improved real-world accuracy of a new model. Custom models do not automatically replace the shipped default model.

See [quick start](native-prototype/training/docs/GETTING_STARTED.md), [Trainer README](native-prototype/training/README.md), [model contribution rules](docs/MODEL_SUBMISSIONS.md), [rights/provenance](RIGHTS_AND_PROVENANCE.md) and [roadmap](ROADMAP.md).

## Contributing and maintenance

CodecCat is maintained in public. Contributions are welcome, including concise bug reports, documentation, tests, native packaging fixes, source-aware accuracy studies and reproducible performance work.

- **Start contributing:** [CONTRIBUTING.md](CONTRIBUTING.md) and [Issue templates](https://github.com/Johnny-Kao/CodecCat/issues/new/choose).
- **Roadmap:** [ROADMAP.md](ROADMAP.md) — shipped work, near-term priorities and longer-term research; no fixed delivery promises.
- **Review and release decisions:** [MAINTAINING.md](MAINTAINING.md), [API stability contract](docs/STABILITY.md), and [Release notes](docs/releases/README.md).
- **Help and conduct:** [SUPPORT.md](SUPPORT.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Please [report security problems privately](SECURITY.md).

You can submit small documentation fixes directly as PRs. Discuss larger API, model, dependency or release changes in an Issue before implementation. CI covers supported platforms; contributors are not expected to reproduce the entire six-platform matrix locally.

## Links

[PyPI](https://pypi.org/project/codeccat/) ·
[GitHub Releases](https://github.com/Johnny-Kao/CodecCat/releases) ·
[Release notes](docs/releases/README.md) ·
[Issues](https://github.com/Johnny-Kao/CodecCat/issues) ·
[Roadmap](ROADMAP.md) ·
[Benchmarks](docs/BENCHMARKS.md) ·
[Stability](docs/STABILITY.md) ·
[License](LICENSE)
