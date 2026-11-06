# Maintainer and decision policy

CodecCat is currently a **maintainer-led OSS project**. The repository owner, [@Johnny-Kao](https://github.com/Johnny-Kao), makes final decisions on scope, merging, security coordination, and releases. This does **not** imply a staffed team or guaranteed response times.

## What the project maintains

- Stable Python public API and CLI for the `0.1.x` line; see [docs/STABILITY.md](docs/STABILITY.md).
- Native inference, supported Python/platform Wheels, fixed-model integrity checks, regression tests and release reproducibility.
- Source-aware accuracy/latency benchmarks, with honest limitations and training-data rights disclosures.
- Community issues, PRs, security concerns and release documentation.

Research experiments may stay unmerged until they are reproducible and useful to released users. The [roadmap](ROADMAP.md) is directional, not a delivery commitment.

## How decisions work

| Change | Expected review |
| --- | --- |
| Typo, documentation, small standalone test | Focused PR; relevant checks |
| Narrow bug fix | Reproducer, before/after behavior, and regression test |
| Performance change | Same-input baseline/candidate measurement, correctness and memory/CPU trade-offs |
| Detection heuristic/model change | Source-stratified errors, known negative controls, fresh independent confirmation source, and compatibility review |
| Public API/dependency/platform/release change | Issue first; backwards-compatibility, provenance and packaging/security impact |
| Security issue | Private coordination per `SECURITY.md` |

### Merge criteria

1. Scope is understandable, with a concrete user-visible benefit or maintainability improvement.
2. No unexplained loss of correct decoded output, API compatibility or release integrity within the affected test scope.
3. Relevant tests or explicit justification for any missing coverage; no obligation to run unrelated expensive workflows for documentation-only changes.
4. External training/test data and attribution are documented; generated model bytes are never silently substituted.
5. A maintainer has reviewed the change. CI passing does not guarantee acceptance.

The maintainer may close stale or out-of-scope proposals with a reason, and can reopen when new evidence appears. Silence is not an approval or a promise.

## Release practice

- Regular contributions target `main` through a PR, not a rewritten tag.
- A release requires a chosen version, an immutable verified tag, supported-platform install checks, model SHA256 matching the expected frozen reference, and documented release notes.
- PyPI publication uses the `pypi` GitHub environment and **OIDC Trusted Publishing**. Publishing is manually initiated, not automatic on PR or push.
- For subsequent versions, bump and review `pyproject.toml`, `native-prototype/Cargo.toml` and [`release-config.json`](release-config.json) together. The [PyPI publisher](.github/workflows/publish-pypi.yml) and [GitHub Release publisher](.github/workflows/publish-stable-release.yml) verify the immutable tag, its exact CI assets and frozen model before manual publication. The first `v0.1.0` release cannot be published again.
- Release notes live in [`docs/releases/`](docs/releases/README.md) and public GitHub Releases. Do not erase historical Alpha limitations in older notes.
- See [docs/PYPI_PUBLISHING.md](docs/PYPI_PUBLISHING.md) and [RIGHTS_AND_PROVENANCE.md](RIGHTS_AND_PROVENANCE.md).

## Maintainer continuity

The CODEOWNERS file routes changes to the current repository owner; it does not block external participation. Additional maintainers may be invited after sustained, trustworthy contributions and an explicit agreement on release/security responsibilities. No additional maintainer positions are currently promised.
