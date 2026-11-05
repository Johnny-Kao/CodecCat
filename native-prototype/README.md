# CodecCat Native Runtime — development source

This directory contains the **single source of truth** for the unreleased
Rust-native CodecCat runtime. The repository root is the distribution build
entry point; do not publish artifacts built from two different project files.

## Runtime and training

- `src/`, `Cargo.toml`: Rust/PyO3 native inference core, CCM1/CCM2 loading.
- `python/codeccat/`: Python public API, CLI and optional lazy legacy shims.
- `training/`: independent `codeccat-train` Python distribution.
- `export_ccm.py`: offline default-model conversion and provenance sidecar.
- `api_parity.py`, `verify_native.py`, `release_reliability.py`:
  installed-wheel regression tests.

The default model is **CCM2 float32**. CCM1 float64 is supported for
compatibility. Default inference dependencies contain no NumPy, SciPy or
scikit-learn. The optional `codeccat[legacy]` extra enables NumPy for
legacy public APIs, while `CodecCat.from_files()` also supports legacy
numeric ZIP models.

## Build

The default fitted CCM model is generated in CI from a pinned corpus and is
**not committed to Git**. A source checkout without that model cannot produce
a usable default-detector wheel. For model-bearing validation, use
`.github/workflows/native-root-package.yml` and
`.github/workflows/native-platform-matrix.yml`. These build from the
repository root and test freshly installed wheels.

The separate trainer can be installed independently:

```sh
python -m pip install ./native-prototype/training
python -m codeccat_train --manifest my-dataset/manifest.jsonl --output my-model.ccm --report my-model-report.json
```

Training manifest entries have paths, encoding labels, group identifiers and
train/validation splits, for example:

```json
{"path":"samples/doc001.bin","encoding":"utf-8","split":"train","group":"source-a"}
```

## Release checks

All six platform wheels, Python 3.10–3.14, default/legacy API parity,
corrupt-model rejection, concurrency, clean installation, model provenance
and distribution authorization must pass **on the final release commit**.
A SHA256 checksum establishes file integrity, not ownership or license.
The `chardet/test-data` files retain individual publishers' copyright;
permission and implications for fitted-model redistribution remain to be
reviewed before official model-bearing release.

See the top-level README for the published Alpha and release status.
