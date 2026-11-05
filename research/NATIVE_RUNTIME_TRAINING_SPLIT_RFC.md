# CodecCat Native Runtime / Training Separation — Architecture RFC v0

Status: **proposed architecture only**. No migration and no change to main,
published Alpha or frozen research accuracy baseline.

## Decision

Build an isolated **Rust core + PyO3/maturin Python extension**, with Python
remaining the public interface, strict decoding/codec interoperability and the
offline training environment. No numpy/scipy/scikit-learn/joblib dependencies
are allowed in the installed inference wheel.

Rust is preferred to C for the combination of native-speed numerical kernels,
memory-safe parsing of untrusted bytes/model files, and maintainable ownership.
**No claim that Rust is faster than C without an actual A/B benchmark.**
Use an abi3 wheel strategy for CPython where supported; verify interpreter
matrix rather than assuming support for all free-threaded builds.

## Verified facts from current `main` (2026-11-05)

- `pyproject.toml`: regular dependencies include `numpy>=2.0` and
  `scikit-learn>=1.4,<2`; `[project.optional-dependencies].train`
  redundantly lists scikit-learn.
- `src/codeccat/usable.py`: standard public `CodecCat()`,
  `detect()`, `decode()`; bundled `models/stable23.zip`;
  `CodecCat.from_files(...)` for custom numeric-only model.
- `src/codeccat/vnext_strong_l2.py`: sklearn TF-IDF + sparse linear score;
  strict-valid candidate filter; first 4096 input bytes used for scoring.
- `src/codeccat/vnext_decoding.py`: strict Python codecs, BOM policy,
  Unicode-text result, ambiguity reporting.
- `src/codeccat/training.py`: a different legacy R21-compatible baseline
  trainer, not the same training artifact as the shipped vNext model.
- Legacy `runtime.py`, `model.py`, `model_io.py` still use numpy.
- Main's wheel workflow trains and bundles stable23; recent 2914/3092
  (94.243%) development result comes from a separate research binary12
  model. **Do not conflate them.**
- The README warns the training corpus and fitted-model redistribution
  provenance must be reviewed before wider distribution.
- Current user API is complete-bytes detection; it is *not* streaming and
  CLI deliberately limits input to 16 MiB.

## Proposed repository boundaries

```text
codeccat/
  pyproject.toml              # maturin backend for inference wheel
  src/codeccat/
    __init__.py               # same public API
    _api.py                   # small Python adapter, dataclasses, codecs
    _model.py                 # resource lookup, version / digest checks
    cli.py                    # existing CLI semantics
    models/default.ccm        # bundled read-only native inference artifact
    py.typed                  # if typed interfaces are maintained
  native/
    Cargo.toml
    src/lib.rs                # Python extension bindings
    src/model.rs              # bounded, defensive model format parser
    src/features.rs           # byte-to-ngram encoding, exact transform
    src/scoring.rs            # sparse accumulation / ranking
    src/structure.rs          # L1 structural byte checks, as proven
  training/
    pyproject.toml            # optional, separately installable
    src/codeccat_train/
      dataset.py              # manifest + provenance + group splits
      fit.py                  # numpy/scikit-learn etc.
      export.py               # portable -> native model conversion
      validate.py             # functional and accuracy parity
    tests/
  tests/
    parity/                   # frozen baseline/candidate outputs
    integration/              # installed wheel + CLI
    performance/              # cold start, warm, memory, throughput
```

If package name splitting is undesirable, offer `pip install codeccat[train]`
with training dependencies, **but training imports must stay lazily isolated**.
A separate `codeccat-train` distribution enforces the boundary more
strongly while allowing the exact same exported model format.

## Python-facing API contract (unchanged)

```python
import codeccat
result = codeccat.detect(raw_bytes)
text = codeccat.decode(raw_bytes)

detector = codeccat.CodecCat()
result = detector.detect(raw_bytes)
custom = codeccat.CodecCat.from_files("custom.ccm")
```

- Keep return fields, exceptions, BOM handling, ambiguity and strict Unicode
  behavior as compatibility contracts.
- Rust takes borrowed input bytes; Python codecs remain authoritative for
  final strict decode of the selected encoding and for codecs lacking native
  equivalents. Minimize copies, not fidelity.
- Core emits ranked candidate codec IDs/scores. Python layer handles
  user-facing objects, code-point decoding and optional language plugin.
- Model loaded lazily exactly once per interpreter with immutable shared data.
- No network access or model download in default import/detect path.
- Training code has no side effects from `import codeccat`.

## Native model format

Build-time export to `default.ccm`, stable versioned schema:

1. Explicit magic / schema major/minor / endian / feature family / numeric
   format / model fingerprint; no pickle.
2. Dense codec-id table and codec aliases; quantization metadata if enabled.
3. Byte-ngram vocabulary, IDF and exact TF-IDF transform parameters.
4. Classifier intercept and learned sparse/dense weights.
5. Integrity checksum and bounded section lengths, checked before indexing.
6. Optional separate L3 artifact; disabled by default unless release
   validation demonstrates benefit.

The artifact must encode all operations required to replicate the chosen
frozen baseline: ngram features, lowercasing, ngram binary/frequency,
sublinear TF, normalization, learned sparse score, tie-break behavior,
top-3 strict validity and BOM policy.

**Do not quantize or sparsify weights during correctness migration.**
First establish exact float64-or-matching numerical parity using a small,
fixed-kernel implementation. Then consider float32/quantization behind an
accuracy gate. The trained model may dominate wheel and memory footprint;
small source-code size does not establish small runtime RAM.

## Experimental migration gates

### Gate 0 — baseline freeze
- Identify **one chosen** public model snapshot and its provenance
  (packaged stable23 vs research binary12). Preserve separate baselines
  for each if compatibility is needed.
- SHA256 each input case; store expected encoding, accepted/ambiguous,
  exact Unicode output, candidate order and normalized BOM result.
- Existing 3092 mixed fixtures are development regression only, not a
  blind confirmation cohort.

### Gate 1 — Rust equivalence prototype
- Implement feature generation and dot product for one frozen model.
- Match TF-IDF and score/ranking within defined floating-point tolerance;
  require **exact decoded output and selected result parity** on all
  frozen regression cases.
- Include malformed, zero-length, BOM, truncated UTF, huge models,
  cross-chunk boundaries and adversarial model-input tests.
- Do not try to increase accuracy at this phase.

### Gate 2 — minimal package
- Wheel built by maturin with compiled Rust module and bundled default
  model; Python 3.10+ per current public API. Verify supported CPython,
  architecture and operating system wheel matrix.
- `pip install codeccat` into fresh venv then assert no pip dependency on
  numpy/scipy/sklearn/joblib, including transitive dependencies.
- Compare file sizes, `import codeccat` latency, lazy model load latency
  and cold/warm detect RSS + CPU vs frozen Python package.
- Validate CLI behavior and optional model loading.

### Gate 3 — train/serve contract
- `python -m codeccat_train fit ...`, `export ...`,
  `validate ...` should accept user-provided bytes and verified labels.
- Split by source-group/document family before any feature selection.
- Produce immutable artifact + training manifest (versions, seed,
  dataset hashes, codec classes, feature params, model digest).
- New model loadable by runtime without importing sklearn or numpy.
- **Do not promote** training-score improvements to release claims
  without independent provenance-verified confirmation.

### Gate 4 — performance and maintenance
- Benchmark with realistic small and long inputs and source/codec mixtures,
  including repeated calls and first-call cold start.
- Track latency p50/p95, cycles/CPU, peak and steady RSS, model file size,
  wheel size, binaries/OS coverage; compare direct C prototype only if Rust
  cost or ABI packaging dominates.
- Only after Gate 1 parity, profile and optimize allocations, representation,
  common scan and candidate validation; require no correctness regression.

## Exclusions and explicit tradeoffs

- **Not** a full Rust rewrite of Python codecs, Python packaging, or the
  training optimizer.
- **Not** a promise of fewer than 100KB of compiled code or that smaller
  source automatically gives faster machine code.
- A native extension requires platform wheels and a build toolchain for
  source installs; without wheels, installation convenience worsens.
- The model's learned weights can occupy more RAM than runtime source or
  binary code. Accuracy-preserving compression needs measurement.
- Legacy API compatibility must be documented: keeping old numpy-backed
  symbols in the default import path would violate zero-dependency policy.
  Plan a deprecated separate compatibility extra/package, not silent breakage.
- Avoid running untrusted model archives through pickle/joblib.

## Recommended first PR (Draft, research/staging only)

A no-production-change Gate-0 snapshot and Gate-1 Rust prototype. No
replacement of the published Alpha until parity and clean wheel regression
are confirmed.
