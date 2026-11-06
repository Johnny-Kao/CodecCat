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
2. Prepare and test the exact Git tag. Build and **install-test** supported Wheels; pin the frozen model and record SHA256 checksums and signed/verified provenance where available.
3. Publish the matching **GitHub Release** and [release notes](releases/README.md), without replacing past tags or binaries.
4. In a reviewed PR, update the **currently first-release-specific** tag, commit, model and asset validation in `publish-pypi.yml`; it is deliberately pinned to `v0.1.0` and **cannot be used for v0.1.1 unmodified**. Do not remove integrity checks merely to make a workflow green.
5. Manually run the workflow from `main` with `publish=false` and inspect the verified-Wheel artifact and Twine metadata check.
6. Confirm the PyPI Trusted Publisher and protected GitHub `pypi` environment, then manually run `publish=true` for the intended new tag.
7. Verify PyPI's project/version JSON contains the expected supported Wheels and matching SHA256, and perform a fresh-environment `pip install codeccat==VERSION` import/decoding smoke test.

The initial release contains **six binary Wheels and no source distribution**. A future sdist must be separately tested to include and load the correct model. PyPI filenames and uploaded versions are immutable; do not try to overwrite a published version.

See [MAINTAINING.md](../MAINTAINING.md) and the [GitHub Release history](https://github.com/Johnny-Kao/CodecCat/releases).
