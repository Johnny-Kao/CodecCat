# PyPI release procedure

The initial distribution is the **already verified** GitHub Stable release `v0.1.0`. The PyPI workflow **does not rebuild** or retrain any binaries: it downloads the six existing release Wheels, checks every source SHA256 and bundled model SHA256, validates metadata, and only then grants the separate publishing job OIDC access.

## PyPI configuration (one-time, account owner)

Because the PyPI project `codeccat` has not been created, configure a **Pending Trusted Publisher** under your account's **Publishing** page: https://pypi.org/manage/account/publishing/

Use:
- PyPI project: `codeccat`
- Publisher: **GitHub**
- Owner: `Johnny-Kao`
- Repository: `CodecCat`
- Workflow filename: `publish-pypi.yml` (basename only)
- GitHub environment: `pypi`

GitHub workflow: `.github/workflows/publish-pypi.yml`. The `pypi` job requires `id-token: write` and no password or token.

The repository's `pypi` environment should be restricted to the `main` branch; add required reviewer protection if available and desired. Do not configure a separate publisher for other workflows.

## First release

1. Run **Publish CodecCat to PyPI** from GitHub Actions, ref `main`, inputs `tag=v0.1.0`, `publish=false`. This verifies all six Wheels without uploading.
2. After PyPI confirms the Pending Trusted Publisher, rerun the same workflow from `main` with `publish=true`.
3. Verify https://pypi.org/project/codeccat/0.1.0/ and install with `python -m pip install codeccat==0.1.0` in a fresh supported platform environment.

No source distribution (sdist) is uploaded in this first release. Unsupported architectures and Linux environments below glibc 2.28 cannot install from these six Wheels. Add an sdist only after explicitly validating its model packaging and build-dependency/reproducibility path.

PyPI distribution filenames are **immutable** after upload; to replace a package later, make a new release version instead of trying to overwrite 0.1.0.

Official instructions: https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
