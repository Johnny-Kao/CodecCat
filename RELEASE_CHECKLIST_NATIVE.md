# Historical Native 0.1.0a3 release gate (2026-11-06)

> **Historical research checklist, not the current release procedure.** Stable v0.1.0 has since shipped to GitHub and PyPI. See [MAINTAINING.md](MAINTAINING.md), [docs/PYPI_PUBLISHING.md](docs/PYPI_PUBLISHING.md), and [docs/releases/](docs/releases/README.md) for the current process.


This checklist applies to the **same immutable commit** that will be tagged.
No GitHub Release or PyPI publication is authorized by CI status alone.

## CI gates

- [ ] Native public API and default-install zero-ML-dependency verification.
- [ ] Python 3.10, 3.11, 3.12, 3.13, 3.14; optional legacy extra.
- [ ] Actual **root-built** model-bearing wheels: Linux x64/ARM64,
      Windows x64/ARM64, macOS Intel/ARM64.
- [ ] Installed-wheel CLI Unicode, corrupt CCM1/CCM2, concurrency, and
      numerical/API parity.
- [ ] Run `python native-prototype/audit_release_wheel.py dist/*.whl` on each final wheel and inspect its report.
- [ ] Verify release assets correspond to the final tag commit and have
      SHA256SUMS. Do not upload artifacts from a different commit.
- [ ] Decide whether a source distribution can include a usable model or
      needs an explicit source-build model-generation process. A bare source
      checkout is currently not a working model-bearing installation.

## Distribution / provenance review

The frozen training corpus is `chardet/test-data` pinned to
`b0c0d206913ba118ffea2f7d306883e7a4a76db7`.
Its license notice states that each test file remains copyrighted by its
publisher. Public availability does **not** establish a blanket license to
redistribute source texts or a fitted model. Check the corpus catalog and
actual training use, record the applicable basis for fitted-parameter
distribution, and document the resulting risk assessment before publishing the binary
default model. See RIGHTS_AND_PROVENANCE.md for the public rights-inquiry
procedure. The contact mechanism is not a liability waiver.

CCM SHA256 and the `model_fingerprint` prove integrity/lineage checks,
**not** a license, authenticity of origin, or distribution permission.

## Release process

1. Keep the working branch `refactor/native-inference-v1` isolated until
   these gates are complete.
2. Review the diff against `main`; avoid committing generated models.
3. After a green final-sha test matrix and provenance approval, integrate
   through a reviewed PR, tag the exact tested commit `v0.1.0a3`, and attach
   the exact matching six platform wheels plus checksums and provenance.
4. Publish a GitHub pre-release (version `0.1.0a3` is an Alpha). Do not
   equate it with a stable release; no PyPI publication is implicit.
5. Verify the release download, installed-wheel import, API and CLI.

Historical `v0.1.0a1` and `v0.1.0a2` releases must not be rewritten.
