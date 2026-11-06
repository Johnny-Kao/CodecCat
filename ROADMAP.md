# CodecCat roadmap

**Status: v0.1.0 Stable published on [PyPI](https://pypi.org/project/codeccat/0.1.0/) and [GitHub Releases](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.0/).**

This is a public, **non-binding** direction of work—not an issue backlog or release-date promise. Proposals should be opened as [Issues](https://github.com/Johnny-Kao/CodecCat/issues/new/choose); scope and priorities can change with evidence and contributor capacity.

## Shipped

- [x] First Stable `0.1.0`, Python 3.10–3.14 public API and CLI contract.
- [x] Native Rust/PyO3 inference; default install does not require NumPy/scikit-learn.
- [x] Six supported Wheels: manylinux glibc 2.28+ x86-64/ARM64, macOS x86-64/ARM64, Windows x86-64/ARM64.
- [x] Immutable-model SHA256 and artifact verification; manual OIDC Trusted Publishing to PyPI.
- [x] Fixed-source reproducible decoding and performance comparison with documented training-source overlap.
- [x] Community contribution, disclosure, review, roadmap and release-note conventions.

## Next: maintenance and distribution

| Workstream | Why it matters | Acceptance evidence |
| --- | --- | --- |
| Source distribution (sdist) and unsupported architectures | PyPI currently provides six binary Wheels; source installs are not a tested fallback | Reproducible source build that includes/verifies the frozen model; isolated install on a non-Wheel target |
| Future-release PyPI automation | First-run publisher currently pins `v0.1.0` and an exact commit | Reviewed parameterized tag/model/asset verification without weakening OIDC, SHA256 or manual approval |
| Documentation and contributor experience | Make first contribution and bug reporting low-friction | Working examples, issue/PR templates, community docs, and docs-link CI |
| Packaging compatibility | Avoid installing an incompatible Wheel or silently missing model data | Tested supported Python/platform matrix, model hash and actual import/decoding |

## Next: quality and performance (research before release)

| Workstream | Goal | Guardrail |
| --- | --- | --- |
| Independent corpus coverage | Separate training/development sets from new language/encoding families | Exact Unicode ground truth, provenance and per-codec outcomes; never call a tuned corpus blind |
| Correctness edge cases | Reduce silent mis-decoding, including UTF-8 versus legacy single-byte ambiguities | Negative controls and same-byte paired regression; no unjustified heuristic override |
| Memory profiling | Explain native model/Python startup/per-request RSS and allocations | Separate baseline interpreter cost; measure peak and steady-state over repetitions |
| Performance profiling | Improve p50/p95 and CPU without hiding correctness losses | Same release API, frozen model, pinned hardware/workload, repeated runs and uncertainty |
| API and packaging ergonomics | Reduce friction while preserving `0.1.x` compatibility | Documented compatibility review and installed-Wheel tests |

## Not currently promised

- 100% encoding accuracy, automatic correction of arbitrary binary data, or unique identification when two encodings produce valid text.
- Streaming detection API, new runtimes, an online hosted service, or a fixed shipping schedule.
- Immediate merger of experimental models purely because of a score on previously inspected fixtures.

A feature becomes committed scope only when maintainers accept a concrete Issue/PR with tests and release implications. See [CONTRIBUTING.md](CONTRIBUTING.md) and [MAINTAINING.md](MAINTAINING.md).
