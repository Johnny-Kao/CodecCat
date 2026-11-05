# CodecCat Native Runtime — Gate 1 implementation

Status: **research/staging prototype**, branch `refactor/native-inference-v1`.
Not merged into `main`; the published Alpha and existing `codeccat.detect`
API are unchanged.

## Completed

- `src/lib.rs`: Rust TF-IDF feature extraction, L2 normalization, native
  sparse weighted linear scores and Python binding (`NativeL2`).
- `src/format.rs`: versioned numeric-only CCM1 model loader with size,
  shape, UTF-8, finite-number and SHA256 integrity validation.
- `export_ccm.py`: training-side-only export of the frozen Alpha
  `stable23.zip` model to `stable23.ccm` (no pickle at inference).
- Separate `pyproject.toml`: clean PyO3/maturin wheel, declaring
  **no Python dependencies**.
- `verify_native.py`: direct tensor constructor and CCM1 loader both match
  the Python/scikit reference numerical scores and class order; corrupt
  artifacts are rejected.

## Verified in GitHub Actions

[Native L2 parity run 42020486468](https://github.com/Johnny-Kao/CodecCat/actions/runs/42020486468):

- Rust wheel built on Ubuntu x86_64, CPython abi3 compatible baseline;
- fresh venv has **no** numpy/scipy/scikit-learn/joblib after installation;
- the frozen Alpha model is trained and exported only in a separate
  training/verification interpreter;
- fresh inference venv loads `stable23.ccm` and produces numerical scores;
- 268 representative/corner input cases match Python L2:
  maximum absolute error `1.0658141036401503e-14`,
  **zero ranking mismatches**;
- CCM1 model length **32,479,080 bytes** (~31.0 MiB), uncompressed.

## Local use (after building a wheel)

```sh
python -m pip install maturin
cd native-prototype
maturin build --release --out dist
python -m pip install dist/*.whl
```

Train + export **in a separate environment containing sklearn**:

```sh
PYTHONPATH=src python native-prototype/export_ccm.py \
  --source src/codeccat/models/stable23.zip \
  --output native-prototype/model/stable23.ccm
```

Inference without NumPy:

```python
from pathlib import Path
from _codeccat_native import NativeL2
model = NativeL2.from_model_bytes(Path("stable23.ccm").read_bytes())
classes = model.class_names()
scores = model.scores(b"hello")
```

## Important limitations / next gates

1. Current binding is an **L2 scorer only**. L1 BOM/escape/structural
   evidence, strict codec validation, L3, confidence/ambiguity and
   end-user Python API are not migrated. Do **not** imply that CodecCat
   end-to-end output parity or 94.24% research benchmark was validated.
2. Parity is against the published **Alpha `stable23`** model; the
   94.24% binary12 research model is distinct and not yet exported or tested.
3. The CCM1 artifact is intentionally float64 and uncompressed. Native
   release packaging should prefer lazy, immutable, read-only loading and
   measure model footprint and peak RSS before any compression or quantization.
4. Final installable `codeccat` Wheel will only be declared zero-heavy-dep
   when its *public* `codeccat.detect/decode/CodecCat` API has parity and
   `pyproject.toml` has been refactored. Current root package still depends
   on NumPy/scikit-learn.
5. A model loader verifies corruption, not data provenance or trustworthiness.
   Downstream model licensing must also be reviewed before wider distribution.
6. Add parity for all frozen test cases and L1/strict decode before replacing
   the existing runtime; develop and test via GitHub Actions only.

## Architecture boundary

```text
Python Offline Training (numpy/sklearn)
       |
       | reproducible numeric export
       v
   stable23.ccm (CCM1, SHA256)
       |
       v
 Rust NativeL2 (no numpy, no sklearn)   [COMPLETED]
       |
       v
 Python L1/strict decode/public API     [NEXT]
```
