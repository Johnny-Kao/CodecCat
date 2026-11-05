# CodecCat v0.1.0a3 — native Alpha (draft release notes)

**Status: not published.** These notes describe the proposed native Alpha,
subject to final asset review and publication.

## What's new

- Native Rust/PyO3 inference with default `CCM2` float32 model and
  `CCM1` float64 loading compatibility.
- Standard Python `CodecCat`, `detect`, `decode` API and CLI;
  legacy interfaces exposed lazily via optional `codeccat[legacy]`.
- No NumPy, SciPy or scikit-learn required by a standard inference install.
- Distinct offline `codeccat-train` source package for custom CCM models.
- Bounded model parsing, SHA256 integrity, Unicode CLI handling,
  concurrency and model corruption regression checks.
- Root-directory Maturin build, platform-specific wheels for Linux,
  macOS, Windows (x86-64 and ARM64).

## Verification and caveats

- The bundled native model is a conversion of the Alpha `stable23` model,
  **not** the separate 94.243% research model.
- Past native-vs-Alpha parity: 6,720 / 6,720 calls. This is **behavioral
  equivalence**, not independently measured general accuracy.
- Actual release assets must be tied to an exact tested commit and verified
  with per-file SHA256 checksums. Six-platform Actions artifacts are not the
  same as a published Release.
- Training corpus provenance and rights inquiry procedure:
  [RIGHTS_AND_PROVENANCE.md](RIGHTS_AND_PROVENANCE.md).
- This is an **experimental Alpha**. No PyPI upload is part of this release.
