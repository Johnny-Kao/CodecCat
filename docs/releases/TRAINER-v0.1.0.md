# codeccat-train 0.1.0 — independent Stable release candidate

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

After PyPI Trusted Publisher registration **and** a successful manual upload:

```bash
python -m pip install codeccat-train==0.1.0
```

From a repository checkout, without PyPI:

```bash
python -m pip install ./native-prototype/training
```

Then follow the [Trainer README](../../native-prototype/training/README.md)
or [quick start](../../native-prototype/training/docs/GETTING_STARTED.md).

## Release authorization gate

This file is release candidate documentation, **not evidence of publication**.
A final Git tag `trainer-v0.1.0` must reference the reviewed commit, the
standalone Wheel and sdist must have successful reproducible CI evidence, and
the maintainer must configure a pending **PyPI Trusted Publisher** for this
*new project* (owner `Johnny-Kao`, repository `CodecCat`, workflow
`publish-trainer-pypi.yml`, GitHub environment `pypi-trainer`) before
running any publishing job. The existing `codeccat` Trusted Publisher does
not authorize uploads to `codeccat-train`.

Never overwrite a published distribution or unilaterally promote the official
default Runtime model.
