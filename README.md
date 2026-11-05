# CodecCat

**CodecCat** detects character encodings and decodes byte sequences using a compact trained model. The next release uses **Rust/PyO3 for native model inference** and Python's strict codecs for decoding.

> **Release status:** [v0.1.0a2](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.0a2) is the latest published experimental Alpha. The native `0.1.0a3` package is undergoing release preparation; **it has not yet been published to GitHub Releases or PyPI**. `pip install codeccat` will not work until PyPI publication.

## Install

For the **currently published Alpha (v0.1.0a2)**, install its model-bearing wheel:

```bash
python -m pip install "https://github.com/Johnny-Kao/CodecCat/releases/download/v0.1.0a2/codeccat-0.1.0a2-py3-none-any.whl"
```

The forthcoming native package requires **Python 3.10+** and a compatible Linux, macOS, or Windows wheel. A future PyPI installation will use `python -m pip install codeccat` once published. The native inference wheel has **no NumPy, SciPy, or scikit-learn runtime dependencies**.

**Source builds:** the default trained CCM model is generated separately and excluded from Git; installing a bare checkout without first preparing the model does **not** give a usable default detector. Maintainers should use the model-bearing GitHub Actions build instead. The canonical native distribution entry point is the **repository-root `pyproject.toml`**, backed by Maturin; `native-prototype/` houses the runtime sources.

## Python API

```python
import codeccat

data = "日本語と café".encode("utf-8")
result = codeccat.detect(data)

print(result.encoding)   # detected label, or None
print(result.text)       # strictly decoded str, or None
print(result.ambiguous)  # whether the encoding label is ambiguous

text = codeccat.decode(data)
detector = codeccat.CodecCat()
custom = codeccat.CodecCat.from_files("my-model.ccm")
```

Legacy numeric `.zip` models remain loadable through `CodecCat.from_files()`. Older `Detector`, `RuntimeBundle` and related Python interfaces require the optional NumPy-dependent `codeccat[legacy]` extra when a compatible distribution is available. Optional language-coherence L3 is explicitly enabled with `CodecCat(use_l3=True)`; it is disabled by default.

The API consumes complete byte sequences, **not streaming fragments**. Encoding detection may be intrinsically ambiguous. Always inspect the returned result rather than assuming a chosen label is certain.

## CLI

```bash
codeccat document.txt
codeccat --json document.txt
codeccat --decode document.txt > decoded.txt
cat document.txt | python -m codeccat --json
```

The CLI accepts inputs of up to **16 MiB**. Decoding is strict and does not silently replace invalid byte sequences. The default recognized-BOM policy is `strip`; `--bom-policy preserve` is available in the CLI, as is `bom_policy="preserve"` in Python.

## Train your own model

The offline trainer is **a separate distribution**, `codeccat-train`. It is not installed with the inference package and requires training-only ML dependencies. Until independently released, install it from the source tree into a separate environment:

```bash
python -m pip install ./native-prototype/training
python -m codeccat_train --manifest my-dataset/manifest.jsonl --output my-model.ccm --report my-model-report.json
```

See [trainer documentation](native-prototype/training/README.md) for the manifest format and dataset isolation rules.

## Architecture, limits, and verification

- **L1:** byte-structure/BOM checks.
- **L2:** trained TF-IDF-style features and linear scoring in Rust, bundled as a `CCM2` float32 model; older `CCM1` float64 is supported.
- **L3:** optional Python language-coherence reranking.
- **No pickle deserialization** for the native CCM model; file checksum validation is not proof of authenticity.

Native regression checks include API parity against the prior Alpha, malformed model rejection, concurrency, fresh-installed-wheel testing, Python 3.10–3.14 compatibility and six operating-system/architecture configurations. These tests do **not** establish general accuracy superiority. Performance numbers depend on workload and hardware; see [benchmark notes](native-prototype/BENCHMARK_AND_PLATFORM_2026-11-05.md).

[Cross-platform native CI](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-platform-matrix.yml) · [Root-wheel CI](https://github.com/Johnny-Kao/CodecCat/actions/workflows/native-root-package.yml) · [Release checklist](RELEASE_CHECKLIST_NATIVE.md)

## License, provenance, and contact

CodecCat's original source code is distributed under the [MIT License](LICENSE). This license does not grant rights to third-party training texts. The experimental default model derives from a pinned `chardet/test-data` revision; no source text files are intended to be distributed in the inference Wheel.

See [model provenance and rights inquiries](RIGHTS_AND_PROVENANCE.md). If you have a rights concern, [contact the maintainer through GitHub Issues](https://github.com/Johnny-Kao/CodecCat/issues/new) without posting confidential evidence publicly.

For bugs or feature requests: [GitHub Issues](https://github.com/Johnny-Kao/CodecCat/issues).
