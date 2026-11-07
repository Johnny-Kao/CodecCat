# codeccat-train 0.1.0 — independently published Stable

**Separate package from `codeccat` Runtime.** The package and version below
refer only to the offline, source-/Wheel-installable training tool.
Do not interpret this as a Runtime upgrade or permission to replace its bundled model.

## Features

- Train local or user-owned encoding-labeled byte corpora with a grouped JSONL
  manifest, strict decoding checks and train/validation separation.
- Export numeric-only `CCM1` (float64, default) and compact `CCM2`
  (float32 weights) model files consumable by the published `codeccat` Runtime.
- Compare models on held-out labeled bytes, showing exact Unicode decoding,
  per-codec results and paired helps/harms (requires `codeccat` for evaluation).
- Produce optional L3 language model artifacts from a train-only manifest.
- Reproduce the original official L2/L3 artifacts when used with the exact pinned
  historical source dataset and environment; **not** a claim that arbitrary
  newly trained models improve quality or are safe to promote.

## Package boundaries

- Project: `codeccat-train`, version `0.1.0`, independent MIT-licensed code.
- The Trainer requires NumPy and scikit-learn and is **never** a dependency
  of the production `codeccat` package.
- No training corpus, trained default model, confidential text or private
  sample content is bundled into the source/Wheel distribution.
- Third-party model training data retain their own rights; historical
  reproducibility is not an unrestricted dataset license.
- Test matrix gates (wheel/sdist, clean environment, Python 3.10–3.14,
  Linux/macOS/Windows and published Runtime interoperability) must pass
  before release approval.

## First-install path

Published using project-specific PyPI OIDC Trusted Publishing on 2026-11-07:

```bash
python -m pip install codeccat-train==0.1.0
```

From a repository checkout, without PyPI:

```bash
python -m pip install ./native-prototype/training
```

Then follow the [Trainer README](../../native-prototype/training/README.md)
or [quick start](../../native-prototype/training/docs/GETTING_STARTED.md).

## Verified release evidence

The immutable [trainer-v0.1.0 tag](https://github.com/Johnny-Kao/CodecCat/tree/trainer-v0.1.0) references reviewed commit `875cd16f9490feb843b7a240ea82dae9b4d28218`. [Nine-platform/Python acceptance CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42367403082) passed, then [OIDC publishing CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42367573901) published the checked Wheel and sdist.

- Wheel `codeccat_train-0.1.0-py3-none-any.whl` SHA256 `780eb798f65c3bce5dd91ce9a2586d61b240212807698d0deda52fab29403c84`.
- sdist `codeccat_train-0.1.0.tar.gz` SHA256 `81ae7d5d05bdbcefb009345cacf79a71eb1026b29895d5e0303369412a6d6fca`.

Both digests were independently retrieved from [PyPI](https://pypi.org/project/codeccat-train/0.1.0/) and match the exact prepublication artifacts. The dedicated `pypi-trainer` GitHub environment and PyPI Trusted Publisher do not grant publishing rights to the separate Runtime package.

Never overwrite a published distribution or unilaterally promote the official
default Runtime model.
