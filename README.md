# CodecCat

**CodecCat** is a Python package for character-encoding detection and strict text decoding. It provides a simple Python API and a command-line tool.

**Status:** [v0.1.0a1 Alpha Pre-release](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.0a1) is available with a trained default model. Code is on `main`; PyPI publication has **not** been configured. Research experiments remain on `staging/initial-package`.

## Install the Alpha

Install the wheel directly from the [GitHub Release](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.0a1):

```bash
python -m pip install "https://github.com/Johnny-Kao/CodecCat/releases/download/v0.1.0a1/codeccat-0.1.0a1-py3-none-any.whl"
```

Alternatively, download the wheel from Releases and install the local file. The Release also provides `SHA256SUMS` and a model-provenance manifest. The [build workflow](https://github.com/Johnny-Kao/CodecCat/actions/workflows/alpha-usable-wheel.yml) verifies the wheel and its clean installation on GitHub Actions.

**Important:** `pip install codeccat` does not work yet because this project has not been published to PyPI. Likewise, `pip install .` from source installs the code without the trained default model; use the model-bearing Release wheel or load a custom model with `CodecCat.from_files(...)`.

Requires Python 3.10 or later. This is an experimental pre-release, not a production deployment claim.

## Python

```python
import codeccat

raw = "日本語と café".encode("utf-8")

result = codeccat.detect(raw)
print(result.encoding)     # inferred encoding label or None
print(result.text)         # strictly decoded text or None
print(result.ambiguous)    # original label may be unidentifiable

text = codeccat.decode(raw)  # str | None
```

Load a specific model instead of the bundled default:

```python
from codeccat import CodecCat

detector = CodecCat.from_files("stable23.zip")
result = detector.detect(b"some bytes")
```

The optional language-coherence L3 layer is disabled by default. Enable it explicitly with `CodecCat(use_l3=True)` when the corresponding L3 model is bundled. It increases conditional latency and has not established an independent net accuracy benefit.

## Command line

```bash
codeccat document.txt
codeccat --json document.txt
codeccat --decode document.txt > decoded.txt
cat document.txt | python -m codeccat --json
```

Detection consumes a **complete byte sequence**, not a streaming fragment. CLI input is limited to 16 MiB. Decoding is strict: invalid selected encodings produce `None`/an error, never silent replacement. The default BOM policy strips an initial recognized BOM; pass `bom_policy="preserve"` in Python or `--bom-policy preserve` on the CLI to retain it.

## Design and limitations

CodecCat combines structural byte checks (L1), a compact trained byte n-gram model (L2), and optional decoded-text coherence (L3). The alpha model supports **88 trained encoding labels**; this is not a guarantee that every input can be identified. Structurally indistinguishable encodings may yield the same text; check `result.ambiguous` and `result.candidates`.

Cross-source accuracy is still under evaluation. A small independent repository benchmark gave CodecCat stable23 **25/34** correctly decoded texts versus **23/34** for chardet 7.6, whereas a separate ICU source previously favored chardet. Neither result supports a claim of general superiority. Native x86-64 and ARM64 inference and a clean installed wheel have been exercised in GitHub Actions.

The alpha's training corpus is pinned to `chardet/test-data`. That repository notes that individual source texts retain their publishers' copyrights; provenance and any constraints on downstream redistribution of the fitted model need review before an official public model release. The GitHub Release is an **experimental Alpha** with a bundled fitted model, not a claim of production suitability or blanket permission to redistribute the underlying training texts. Model provenance should be reviewed before wider PyPI or stable-release distribution.

## Development

- `src/codeccat/`: runtime, public API and CLI
- `experiments/logic_first/`: reproducible alpha model build
- `tests/`: decoding and model-contract tests
- `.github/workflows/alpha-usable-wheel.yml`: model build, wheel, fresh environment install and smoke tests
- `staging/initial-package`: historical research and experimental comparisons

The legacy `Detector(RuntimeBundle)` interfaces remain available separately from `CodecCat()`. See the build workflow for the exact training source revision and model provenance.
