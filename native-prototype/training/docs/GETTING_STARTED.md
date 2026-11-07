# Offline model training with codeccat-train

CodecCat offers **one** offline training tool for anyone's dataset. The CLI and model format are shared; official and user-owned datasets differ in provenance and what is allowed to ship as an official release.

**Release candidate (2026-11-07):** Offline training and candidate evaluation are E2E-verified against the published Runtime. The source tree is preparing independent `codeccat-train==0.1.0` Stable distribution, which must pass release CI and PyPI Trusted Publishing before it becomes publicly installable from PyPI. Exact historical official L2 CCM2 and optional L3 replay have passed under pinned sources and dependencies; this is not data-rights clearance, stronger real-world accuracy, or authorization to promote a new default model. The separately published Runtime remains `codeccat==0.1.0`.

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

The default custom-model output is `CCM1`; use `--precision f32` for official-compatible `CCM2`. The pinned historical recipe has matched the official CCM2 bytes, but arbitrary custom datasets or versions do not guarantee that result.

## Manifest format

Each line of `manifest.jsonl` is a JSON object:

```jsonl
{"path":"samples/document-001.bin","encoding":"utf-8","split":"train","group":"document-001"}
{"path":"samples/document-002.bin","encoding":"cp1252","split":"validation","group":"document-002"}
```

Paths are relative to the manifest directory. `group` identifies a document/source that must not cross the training/validation boundary. The loader rejects escaping paths, duplicate bytes with conflicting labels/splits, unknown codecs and non-decodable labeled bytes. **This is only a first barrier:** near-duplicates, source overlap and corpus licenses remain the user's responsibility.

Use bytes with independently known encodings and valid redistribution/training rights. Don't upload raw confidential files or training data to GitHub Actions.

## Acceptance and next steps

- **P0:** [Trainer E2E CI](../../../.github/workflows/trainer-e2e.yml) installs Trainer separately, trains twice and checks same-env reproducibility, then loads the resulting model with an independent **PyPI `codeccat==0.1.0`** Runtime environment; it also tests some unsafe manifest cases. No model is promoted by this check.
- **P1:** Pinned historical input manifest, exact L2 residual training, and optional unfiltered L3 replay are implemented and have matched frozen reference artifacts. Third-party corpus rights review and independent source-overlap assurance are not complete.
- **P2:** Candidate-vs-Stable comparison and model submission checks exist; exact historical reproduction is guarded separately from any future model-quality or release approval. No independent candidate release is implied.

See [RIGHTS_AND_PROVENANCE.md](../../../RIGHTS_AND_PROVENANCE.md), [ROADMAP.md](../../../ROADMAP.md), and [CONTRIBUTING.md](../../../CONTRIBUTING.md).

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

The evaluator scans all manifest splits, rejecting identical bytes and source groups reused across train, validation and test. This prevents exact within-manifest leakage, but it cannot identify near-duplicates or overlap with a separate training manifest. See the [official dataset provenance status](../../../datasets/official/README.md) before claiming official-model reproducibility.

The evaluation manifest uses the same JSONL keys with `"split":"test"` (or explicitly `--split validation` for development tests). Never present validation or previously used training sources as blind independent test data. Results are descriptive, not an automatic official-model upgrade.

## Compact numeric model output

By default, `codeccat-train fit` writes `CCM1` (float64 weights). Supply `--precision f32` to write a `CCM2` model with float32 weights, compatible with the published Runtime:

```bash
codeccat-train fit --manifest my-data/manifest.jsonl --output custom-f32.ccm \
  --report training-f32.json --precision f32
```

This changes only stored weight precision and may introduce minor numerical differences. Reproducing the official model additionally requires the pinned dataset manifest, `--train-only`, `--residual-focus` and verified environment, as shown by the historical research workflow.

## Replaying historical residual-focused training (research only)

For investigations that need to match the old Stable23 data-selection
method, `--residual-focus` applies the historical sample filter before the
**same** L2 fitting core. `--train-only` accepts a manifest with only
`"split":"train"` records, which matches the old release-building choice
to fit on every eligible example:

```bash
codeccat-train fit --manifest local/training.jsonl --train-only \
  --residual-focus --precision f32 --output candidate.ccm \
  --report candidate-report.json
```

In train-only mode, `validation_codec_label_accuracy` is `null` and no
independent accuracy claim is made. The report records original training
size, selected training size, removed counts per codec and dataset
fingerprints before and after filtering. The default community path still
**requires** validation records and does not filter samples.

The full pinned-source replay has reproduced the existing L2 model exactly. The separate `fit-l3` command handles the optional, unfiltered L3 model and also matched the frozen reference. See [official reconstruction status](../../../datasets/official/README.md).


## Training optional L3 (off by default)

`fit-l3` uses the full, unfiltered training rows (not the residual L2 subset), preserving the historical Unicode-trigram statistics:

```bash
codeccat-train fit-l3 --manifest /path/to/train-only.jsonl \
  --output /path/to/language-l3.json \
  --report /path/to/l3-report.json
```

The [manual L2 + L3 reproduction gate](../../../.github/workflows/official-stable23-replay.yml) compares both generated artifacts against pinned released reference SHA256 values and fails if a required identity check differs. L3 stays disabled by default. Reproduction does **not** establish rights clearance or production accuracy improvements.
