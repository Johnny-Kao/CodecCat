# Contributing to CodecCat

CodecCat welcomes bug reports, documentation improvements, reproducible performance research, tests, and focused code changes. You do not need to ask permission to fix a typo or report a bug.

**Start here:** [README](README.md) · [Roadmap](ROADMAP.md) · [Maintenance policy](MAINTAINING.md) · [API contract](docs/STABILITY.md) · [Code of conduct](CODE_OF_CONDUCT.md)

## Choose the smallest useful contribution

- **Bug or incorrect decoded text:** [open a bug report](https://github.com/Johnny-Kao/CodecCat/issues/new/choose). Include version, platform, Python version, source encoding (if independently known), reproducer and actual/expected results.
- **Feature or public API change:** [open a proposal first](https://github.com/Johnny-Kao/CodecCat/issues/new/choose), especially for breaking changes, new dependencies, new model families, or new benchmark datasets.
- **Small fix, test, or documentation:** a focused Pull Request is welcome without a prior issue.
- **Performance improvement:** provide the exact input characteristics, run environment, baseline/candidate commands, repetitions and output correctness. A speedup that changes decoded text is not a pure performance improvement.
- **Trained models:** read [Model submissions](docs/MODEL_SUBMISSIONS.md) before proposing a model change; raw source data and unofficial models are never automatically promoted.
- **Security or sensitive data:** follow [SECURITY.md](SECURITY.md); do not publish exploit details or private files in an Issue.

A maintainer may ask to narrow scope, request stronger evidence, defer, or decline a change. An Issue is a discussion, not advance approval.

## Prepare a reproducible change

1. Fork the repository; create a branch from the current `main`. Keep unrelated changes separate.
2. Pin source revisions used in tests, and describe the origin and licensing of any fixtures. Prefer minimal synthetic regression bytes over copying entire third-party corpora.
3. Use an installed published Wheel to reproduce a user-facing bug first, then validate the proposed change against the same bytes.
4. Keep tests for the changed path. API or model changes need compatibility and regression checks; portability changes need installed-Wheel checks.
5. Open a Pull Request using the repository template. State **what changed, why, how it was tested, user-visible risks**, and any unresolved limitations.

Some native build tests require a verified frozen model/reference archive and Rust toolchain. The project CI runs Linux, macOS and Windows matrices. Contributors **do not** need to run all six platform builders locally; report which local checks ran and let CI cover supported platforms. See [MAINTAINING.md](MAINTAINING.md) for merge criteria.

## Technical guidelines

- Preserve the [0.1.x public API compatibility contract](docs/STABILITY.md). Avoid adding mandatory ML packages to the default install.
- Prefer exact Unicode output over only matching an encoding label. A valid codec label does **not** prove that the decoded text is correct.
- Compare baseline and candidate using the same input bytes, shipped inference path and bundled model. Distinguish training/development overlap from genuinely untouched confirmation sources.
- Pin external corpora and acknowledge dataset rights. Never commit raw proprietary samples, personal files, secrets, pre-trained model replacements or generated assets without an explicit review of provenance.
- Do not change release artifacts or tags to rewrite history. Versions uploaded to PyPI cannot be overwritten.
- Keep PRs reviewable. Explain any AI-assisted code or generated tests in the PR description, and confirm that you personally checked the proposed changes and evidence. **AI use is welcome; unverified output is not.**
- Be respectful of contributors and maintainers. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Local first steps

```bash
python -m pip install codeccat
python -c "import codeccat; print(codeccat.detect(b'hello').text)"
```

For repository development, inspect `pyproject.toml` and `native-prototype/`. Native source builds need Rust, maturin, and a verified model asset; **a fresh source checkout alone is not guaranteed to build a functional model-bearing Wheel**. Use the GitHub Actions workflows as the source of truth for cross-platform release builds.

We cannot promise a review turnaround or acceptance. Maintainer attention is limited, and a small, well-reproduced change is usually easiest to review.
