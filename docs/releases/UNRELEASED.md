# Released changes in Runtime v0.1.1

**Published 2026-11-07:** The Runtime changes below are available in [codeccat 0.1.1](https://pypi.org/project/codeccat/0.1.1/), released from immutable [v0.1.1](https://github.com/Johnny-Kao/CodecCat/releases/tag/v0.1.1) after SHA256-verified GitHub Release/PyPI Trusted Publishing. The original 0.1.0 assets remain unchanged.

## Runtime source changes

- Added opt-in `codeccat.diagnostics.inspect_ambiguity(original_bytes, result)`, returning explanatory metadata about ambiguous C1-control interpretations among already ranked codecs.
- The diagnostic does **not** replace text or codec decisions, does not mutate the `Detection` result, and is not a calibrated prediction of whether decoding is wrong.
- Kept original `codeccat.detect()`, `codeccat.decode()`, BOM, default model, and L3 opt-in behavior intact.
- Reduced extra diagnostic scan cost on long documents by using compiled regular-expression search; checked the same original historical HTML corpus before/after without changing output.
- Verified Linux/macOS/Windows x64 and ARM64 clean native Wheels, Python 3.10–3.14, model identity, existing API contracts, build/install and source-distribution dry runs. See [PR #47](https://github.com/Johnny-Kao/CodecCat/pull/47), [six-platform CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42343752787) and [validation methodology](../../research/C1_STAGING_VALIDATION_2026-11-07.md).

## Offline training tool (separate package)

- `codeccat-train 0.1.0` has subsequently been **independently published** on [PyPI](https://pypi.org/project/codeccat-train/0.1.0/) from `trainer-v0.1.0`, with a separate OIDC Trusted Publisher. This does **not** publish the Runtime's optional C1 diagnostic or update the Runtime 0.1.0 version. The Trainer supports private labeled data, CCM1/CCM2, candidate evaluation, and optional L3.
- Historical L2 and L3 model replays were reproduced under pinned environments, but no newly trained candidate has been approved to replace the existing official default model.
- Independent Trainer distribution and its release checks are complete. Source-provenance/user-data permissions and future official-model quality gates remain separate from the Runtime patch release.

## Release decision

**Runtime 0.1.1 has been published** with the backward-compatible optional diagnostic. Six platform Wheels and the sdist were assembled from three successful exact-tag workflows and uploaded to PyPI unchanged using OIDC. See [v0.1.1 release notes](v0.1.1.md). Further API/model work requires its own release decision.

