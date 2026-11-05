# CodecCat Native Runtime — staging integration

**Branch:** `refactor/native-inference-v1` (isolated, not merged).
**Status:** Public API, optional L3 and independent user training validated in
GitHub Actions. The published Alpha and `main` remain unchanged.

## Architecture

```text
Offline training environment                 User-facing runtime wheel
  codeccat-train (Python)                        codeccat (Python + Rust)
  NumPy, sklearn                                zero NumPy/sklearn deps
    |                                             |
  custom.ccm ---------------------------------> NativeL2 in Rust
    |                                             |
  numeric model export                          L1 checks + Python codecs
                                                  |
                                             detect / decode / CLI
```

**Only the learned model parameters and the frozen feature transform are
needed for inference.** Native Rust performs the trained TF-IDF transform and
linear scoring; Python's built-in strict codec implementation remains the
authority for Unicode decoding. No pickle is allowed in native model loading.

A default `models/default.ccm` is bundled into the wheel; optional
`language-l3.json` supports `CodecCat(use_l3=True)`. The default remains
L3-disabled to preserve existing behavior.

## Native Python API

```python
import codeccat
out = codeccat.detect(b"hello")
print(out.encoding, out.text, out.ambiguous)

custom = codeccat.CodecCat.from_files("custom.ccm")
out = custom.detect(b"hello")
```

Existing `codeccat.detect()`, `decode()`, `CodecCat()`, `from_files()`
and CLI shape are retained. Legacy R21 `Detector` and `RuntimeBundle` are
not included in this staging wheel: migration/deprecation of that older API
is still an explicit release decision, not silently promised compatibility.

The `from_files` model argument now refers to native CCM1 instead of a
sklearn-era `.zip` artifact. The file-type migration must be documented before
shipping publicly.

## Independent user-data trainer

Only install the heavy trainer into a **separate training environment**:

```sh
python -m pip install ./native-prototype/training
python -m codeccat_train \
  --manifest my-dataset/manifest.jsonl \
  --output my-model.ccm \
  --report my-model-report.json
```

Each line of `my-dataset/manifest.jsonl`:

```json
{"path":"samples/doc001.bin","encoding":"utf-8","split":"train","group":"source-a"}
```

The trainer:

- Reads raw bytes, requires strict decodability under the declared label.
- Rejects absolute/traversal paths, conflicting duplicate bytes,
  train/validation group overlap and unseen validation labels.
- Uses frozen `stable23` feature recipe (2–3 character n-grams,
  sublinear TF-IDF, an SGD modified-Huber linear classifier).
- Emits a versioned numeric-only CCM1 model with SHA256 integrity and
  a provenance/validation report. Groups isolate held-out validation data.
- Does **not** run at normal inference and is not a dependency of the
  native public wheel.

Training requires genuine varied real-world input. The provided tiny
two-codec fixture is exclusively a **smoke test of export/load**, not
a statistically meaningful accuracy estimate.

## Verified CI

Workflow: `.github/workflows/native-public-api.yml`

- Train original Alpha `stable23` model and optional L3 from pinned training
  corpus, generate frozen reference signatures without touching test output.
- Export to CCM1, build Linux x86_64 CPython abi3 native Wheel.
- Install into an isolated interpreter without NumPy/SciPy/sklearn/joblib.
- Compare default + opt-in L3 public API against original Alpha across 6,720
  complete calls, including BOM policy alternatives, Unicode text SHA256,
  candidate order, ambiguity, decision reasons and strict decoding.
- Independently compare native scores to 268 NumPy/sklearn reference score
  vectors (numerical tolerance 1e-9); reject invalid CCM1 checksum.
- Train a custom user model with exactly 2 classes in a separate training
  environment and verify native `CodecCat.from_files` can load it.
- Measure Linux runner per-call CPU/wall time, cold model load and peak RSS.

Reference green run for L3 + user training:
[Actions #42031209505](https://github.com/Johnny-Kao/CodecCat/actions/runs/42031209505).
The unified numeric-parity step was added afterward and requires its own
successful final CI result.

### Observed previous Linux x86_64 results (not other platforms)

A representative run observed 4 KB Alpha detection at 2.99 ms p50 versus
Native 0.322 ms p50 (about 9.3x faster), and max RSS approximately
212 MiB versus 87 MiB. The 32,479,080-byte uncompressed CCM1 default model
dominates the packed wheel (approximately 13 MiB compressed).
These results are machine-, load- and workload-specific.

## Remaining release gates

1. **Cross-platform Wheels:** run Linux, macOS, Windows, Intel/Arm where
   supported; CPython 3.10+ compatibility and common pip installation.
2. **Model provenance:** review redistribution rights of model trained on
   external fixture sources before wider publication.
3. **Research model parity:** separately export/validate the 94.24% binary12
   research classifier; current successful parity is on published Alpha
   stable23, not that research model.
4. **Memory/performance:** profile allocator and model layout before
   float32/quantization; preserve exact classification output.
5. **API migration:** decide legacy R21 symbol compatibility separately.
6. **Security/reliability:** malformed CCM1/fuzzing, concurrent calls,
   packaging metadata and wheel repair/audit across platforms.

Do not merge this branch or publish a new release without the required
platform, accuracy, compatibility and provenance gates.
