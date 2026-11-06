# PyPI release and trusted publisher procedure

**Current status:** CodecCat **v0.1.0 is published** on [PyPI](https://pypi.org/project/codeccat/0.1.0/). The first upload used GitHub Actions **OIDC Trusted Publishing**, verified against the six original Stable GitHub Release Wheels with matching SHA256 and frozen-model digest. [Successful publication run](https://github.com/Johnny-Kao/CodecCat/actions/runs/42167721009).

## Current publisher configuration

- PyPI project: `codeccat`
- GitHub owner: `Johnny-Kao`
- Repository: `CodecCat`
- Workflow filename: `publish-pypi.yml`
- Environment: `pypi`
- Workflow: [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml)
- Authentication: environment-scoped OIDC (`id-token: write`) in the **publishing job only**; no API Token in repo secrets.

The environment should be restricted to trusted release branches or explicitly approved deployment. Changes to publish permissions, model provenance and release verification require maintainer review.

## Procedure for a future release

1. Choose a new semantic version and review compatibility, user-visible changes, rights/provenance, and [Stable policy](STABILITY.md).
2. Prepare an immutable Git tag. Tag pushes trigger **six-platform Wheels**, **manylinux** and the **sdist round-trip** workflow; collect the three successful run IDs, all built from that exact tag Commit. The source distribution must include the same numeric model and successfully rebuild/install.
3. Add `docs/releases/vVERSION.md` **before tagging**. Manually run `publish-stable-release.yml` with the tag and three verified run IDs, first with `publish=false` for a no-upload assembly and then `publish=true` to create the matching **GitHub Release**; no replacement of old tags or binaries.
4. In a reviewed PR, update `pyproject.toml`, `native-prototype/Cargo.toml` and **`release-config.json`** to the same new version; commit approved model SHA256 and full source attribution in the tag. Keep the release configuration immutable after tagging. The historical `v0.1.0` workflow is intentionally not reusable.
5. Manually run `publish-pypi.yml` from `main` with the new tag and `publish=false`. Its independent verifier checks the Git tag, versioned release configuration, SHA256SUMS, six Wheels, optional sdist, metadata and model SHA256. Do not build replacement Wheels in this step.
6. Confirm the PyPI Trusted Publisher and protected GitHub `pypi` environment, then manually run `publish=true` for the intended new tag.
7. Verify PyPI's project/version JSON contains the expected supported Wheels and matching SHA256, and perform a fresh-environment `pip install codeccat==VERSION` import/decoding smoke test.

The initial `v0.1.0` release contains **six binary Wheels and no sdist**. The subsequent-release workflow requires a tested sdist in the GitHub Release. `source-distribution.yml` verifies source -> Wheel round-trip, and a five-version Python compatibility matrix. Original third-party training text is never bundled. Published PyPI artifacts are immutable and cannot be overwritten.

See [MAINTAINING.md](../MAINTAINING.md) and the [GitHub Release history](https://github.com/Johnny-Kao/CodecCat/releases).
