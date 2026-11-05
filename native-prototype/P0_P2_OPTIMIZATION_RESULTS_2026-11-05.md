# CodecCat Native P0–P2 Memory Optimization — 2026-11-05

Status: **validated in staging**, branch `refactor/native-inference-v1`.
No change to `main` or published Alpha. This is the packaged Alpha stable23
model, **not** the separate 94.243%-benchmark binary12 research model.

## Changes implemented

### P0 — model loading and tensor layout
- Python `TrainableDetector.load_native` now calls Rust
  `NativeL2.from_model_file` instead of `Path.read_bytes` twice.
- Rust uses a read-only memory-mapped file for the bounded, SHA256-checked
  parser; no full-file Python bytes allocation. The map is released after
  parsing. Model data remains in owned immutable Rust vectors, not a lifetime-
  dependent borrowed mmap pointer.
- Replaced `Vec<Vec<f64>>` with contiguous class-major weights, keeping
  trained weight indexing and accumulation ordering unchanged.

### P1 — floating-point precision
- Introduced optional versioned **CCM2** serialization: weight matrix in
  float32; IDF, classifier intercept and f64 score accumulation remain
  float64. Existing **CCM1** float64 remains supported.
- Promoted CCM2 to the *staging* default only after all 6720 API signatures
  matched the pinned Alpha model, with optional L3 and both BOM policies.
  The numeric f64 reference comparison uses a separate original CCM1 file,
  not the reduced-precision artifact.
- Individual confidence scores are not claimed bitwise-identical to an
  original float64 model. The public decision and decoded Unicode output
  comparison was exact on the locked corpus.

### P2 — memory-efficient vocabulary
- Replaced Rust's allocating `HashMap<u64,usize>` dictionary with a compact
  contiguous open-addressed table of packed byte-ngram keys and feature
  indices.
- Lookup is expected O(1) with bounded load factor. Eliminated much of the
  allocation overhead without changing string normalization or feature IDs.
- A previously explored sorted-vector binary-search table saved memory but
  slowed 4KB inference; it was replaced rather than retained.

## Verified CI

- [Final native public API and f32 experiment](https://github.com/Johnny-Kao/CodecCat/actions/runs/42052538603) — PASS.
- [Final six-platform x86-64 / ARM64 matrix](https://github.com/Johnny-Kao/CodecCat/actions/runs/42052569701) — PASS on all six.
- [Final competitor resource comparison](https://github.com/Johnny-Kao/CodecCat/actions/runs/42052594947) — PASS.

**Parity:**
- Full public API: 6720/6720 exact output comparisons, including L3 and BOM modes.
- Original float64 L2 score parity: 267 cases, max abs score error
  `1.0658141036401503e-14`, zero rank mismatches.
- CCM2 float32 complete API experimental comparison: 6720, **0 different**.
- Six platform wheels successfully install with no NumPy, SciPy, sklearn
  or joblib and pass the same fixed API / numerical regression checks.
- External user-trained binary-class CCM1 remains loadable by the same Rust
  native runtime.

## Memory and performance

Example standalone public API CI on same Linux x86-64 runner:

| Metric | Original Alpha (NumPy/sklearn) | Optimized Native CCM2 |
|---|---:|---:|
| 4KB detection p50 | 2.4429 ms | 0.1908 ms |
| 4KB detection p95 | 2.5649 ms | 0.2018 ms |
| Model load | 838.5 ms | 41.3 ms |
| Peak process RSS | 212.7 MiB | 54.7 MiB |
| Uncompressed model artifact | CCM1 32.48 MB | CCM2 16.64 MB |

The sample and host influence timing and RSS. Source-code size is unrelated
to parameter size or baseline Python interpreter memory. The P0–P2 optimized
CCM2 process still consumes memory for the interpreter, Rust extension,
immutable model parameters, and per-call feature buffers.

Final 304-sample same-run resource benchmark (one Linux runner,
different isolated Python venv for each library):

| Metric | Native CCM2 | Original Alpha | chardet 7.5.0 | charset-normalizer 3.5.2 |
|---|---:|---:|---:|---:|
| Exact decoded Unicode | 294/304 | 294/304 | 286/304 | 249/304 |
| p50 detection latency | 0.4712 ms | 1.4649 ms | 0.5388 ms | 0.8903 ms |
| p95 detection latency | 1.4944 ms | 4.6826 ms | 3.6182 ms | 2.7654 ms |
| Total process CPU | 182.95 ms | 609.78 ms | 314.61 ms | 347.20 ms |
| Peak RSS | 55.12 MiB | 211.71 MiB | 51.59 MiB | 25.88 MiB |
| Isolated env disk size | 30 MB | 278 MB | 16 MB | 14 MB |

**Caveats:**
- The 304 cases come from a previously used development fixture and edge
  examples. Their 96.71% Native result is **not** a blinded generalization
  accuracy claim, not equivalent to the 94.24% research five-panel model,
  and may vary between retrained Alpha builds.
- Different detector public APIs perform different stages. CodecCat performs
  strict decoding within the measured call; chardet returns only a codec label.
  For correctness, common strict decoding was performed *after* timing.
- Measurements are specific to one Linux x86-64 CI runner and installed
  versions. Six-platform CI checks output correctness, not equal performance.
- Float32 may change extremely close rankings on different unseen bytes;
  exact regression parity is necessary but not a proof of all-input parity.

## Trade-offs / decision

All requested P0–P2 tasks are **implemented and validated** in staging.
The new staging default is CCM2 float32, with CCM1 compatibility retained.
The risk of lossy numeric precision is accepted **only within this staging
branch**, backed by the exact fixed-output regression and six-platform gate.

Further compression options, requiring separate approval/validation:
- sparse/blocked or compressed tensor storage (might reduce wheel size);
- smaller ngram vocabulary or re-training (affects model behavior);
- lower than 32-bit quantization (inference drift risk).

Do not continue into these optimization families automatically.
Cross-platform output equivalence is not a publication or licensing approval.
