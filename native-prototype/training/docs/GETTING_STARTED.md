# Offline model training (experimental)

CodecCat offers **one** offline training tool for anyone's dataset. The CLI and model format are shared; official and user-owned datasets differ in provenance and what is allowed to ship as an official release.

**Status:** Source-distributed trainer prototype `codeccat-train==0.1.0a3`. It is **not yet published on PyPI**, and does not yet reproduce the entire official Stable training pipeline. Public Native `codeccat==0.1.0` remains unchanged.

## Quick local example

Use Python 3.12 and keep the trainer out of production environments:

```bash
python -m venv .trainer-env
.trainer-env/bin/python -m pip install ./native-prototype/training
.trainer-env/bin/python native-prototype/training/make_smoke_data.py
.trainer-env/bin/python -m codeccat_train \
  --manifest native-prototype/training/smoke-data/manifest.jsonl \
  --output custom.ccm \
  --report training-report.json
```

The synthetic smoke dataset consists of UTF-8 and Windows-1252 bytes. Its purpose is checking the **pipeline**, not measuring real-world encoding accuracy. Do not use its reported validation score to claim production quality.

A separate environment can install and use the released Runtime:

```bash
python -m venv .runtime-env
.runtime-env/bin/python -m pip install --only-binary=:all: codeccat==0.1.0
.runtime-env/bin/python - <<'PY'
from codeccat import CodecCat
model = CodecCat.from_files("custom.ccm")
result = model.detect("日本語".encode("utf-8"))
print(result.encoding, result.text, result.ambiguous)
PY
```

The model output uses `CCM1`; the current official bundled model uses `CCM2`. The ability to load a newly trained CCM1 model is **not** a guarantee of identical inference quality or reproducibility of the official model.

## Manifest format

Each line of `manifest.jsonl` is a JSON object:

```jsonl
{"path":"samples/document-001.bin","encoding":"utf-8","split":"train","group":"document-001"}
{"path":"samples/document-002.bin","encoding":"cp1252","split":"validation","group":"document-002"}
```

Paths are relative to the manifest directory. `group` identifies a document/source that must not cross the training/validation boundary. The loader rejects escaping paths, duplicate bytes with conflicting labels/splits, unknown codecs and non-decodable labeled bytes. **This is only a first barrier:** near-duplicates, source overlap and corpus licenses remain the user's responsibility.

Use bytes with independently known encodings and valid redistribution/training rights. Don't upload raw confidential files or training data to GitHub Actions.

## Acceptance and next steps

- **P0:** [Trainer E2E CI](../../.github/workflows/trainer-e2e.yml) installs Trainer separately, trains twice and checks same-env reproducibility, then loads the resulting model with an independent **PyPI `codeccat==0.1.0`** Runtime environment; it also tests some unsafe manifest cases. No model is promoted by this check.
- **P1:** Add official dataset source manifest and explicit license/provenance assessment, holdout/evaluation by exact decoded text, per-codec deltas, and frozen training parameters. No claim that every source can legally be mirrored in this repo.
- **P2:** Model comparison and contributor submission gate, independent candidate model release, and release process separated from Stable inference.

See [RIGHTS_AND_PROVENANCE.md](../../RIGHTS_AND_PROVENANCE.md), [ROADMAP.md](../../ROADMAP.md), and [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Compare your model against official Stable

Install the public Runtime in the **evaluation environment** (not required for the training-only environment), then run:

```bash
codeccat-train evaluate \
  --model custom.ccm \
  --manifest my-data/test-manifest.jsonl \
  --split test \
  --report comparison.json
```

The `evaluation` command compares candidate and bundled Stable on identical bytes, reporting **correct decoded Unicode text**, correct codec labels, per-codec accuracy, and how many examples improve or regress. No sample text or raw bytes appear in the output. Optionally use `--baseline-model /path/to/another.ccm`.

The evaluation manifest uses the same JSONL keys with `"split":"test"` (or explicitly `--split validation` for development tests). Never present validation or previously used training sources as blind independent test data. Results are descriptive, not an automatic official-model upgrade.
