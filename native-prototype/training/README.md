# codeccat-train (offline only)

Independent Python distribution for training and exporting CodecCat native
`.ccm` models. **Not a runtime dependency** of `codeccat`. The train
environment requires NumPy and scikit-learn; production inference does not.

**Implementation status (2026-11-07):** The core offline Trainer and end-to-end
published-Runtime model-loading workflow are implemented and verified.
The trainer remains a **source-installable Alpha (`0.1.0a3`)**, not an
independently published stable PyPI package. Its historical official L2/L3
reproduction tests establish *reproducibility*, not a better new model or
permission to publish third-party training texts.

The same Trainer is used for both workflows: private community-supplied datasets
remain the user's responsibility; the project-maintained dataset is governed by
a separate rights and default-model acceptance decision. No user dataset or
generated model is automatically submitted to the CodecCat project.

From a checkout (experimental source distribution; not on PyPI):

```bash
python -m pip install ./native-prototype/training
python -m codeccat_train fit --manifest dataset/manifest.jsonl --output custom.ccm --report training-report.json
```

Dataset manifest format (one JSON object per line, paths relative to the
manifest file):

```json
{"path":"samples/a.bin","encoding":"utf-8","split":"train","group":"document-a"}
{"path":"samples/b.bin","encoding":"shift_jis","split":"validation","group":"document-b"}
```

The trainer checks strict decodability, rejects path traversal, conflicting
duplicate contents and train/validation source-group overlap. Supply varied,
lawfully obtained text samples with independently known encoding labels.
The training report stores both `manifest_sha256` (the manifest file bytes) and
`dataset_content_fingerprint_sha256` (a deterministic digest of accepted
input-file hashes, codec labels, group IDs and split assignments). Unlike a
manifest hash alone, the second value changes when an input file's content
changes at the same path. This does **not** prove dataset licensing or
independent holdout status, and does not expose raw training text.

The generated report describes model provenance and validation; it is **not**
a guarantee of accuracy on arbitrary real-world text.

See the [CodecCat repository](https://github.com/Johnny-Kao/CodecCat)
for the native runtime and [rights and provenance policy](https://github.com/Johnny-Kao/CodecCat/blob/main/RIGHTS_AND_PROVENANCE.md).

## Verification and release boundary

- **Implemented and CI-verified:** grouped source-manifest ingestion and isolation,
  deterministic same-environment CCM1/CCM2 training, native Runtime loading,
  source-content fingerprints, candidate-vs-Stable Unicode comparison,
  optional L3 training, exact pinned official historical L2/L3 replay.
- **Platform evidence:** training and Runtime interoperability checked on Linux
  x64/ARM64 and macOS ARM64; custom-model Runtime portability checked on
  macOS Intel and Windows x64.
- **Not yet a Stable Trainer release:** no separate `codeccat-train` stable PyPI
  publication or independent release/rollback process. Historical training
  provenance is documented with residual risk, not universally cleared.
- **Not an automatic Runtime update:** installing or training with this tool
  does not alter `codeccat` or the bundled Stable model. A candidate default
  requires a separately approved evaluation and release.

CI evidence: [Trainer E2E](https://github.com/Johnny-Kao/CodecCat/actions/runs/42240463787),
[macOS ARM64 native trainer](https://github.com/Johnny-Kao/CodecCat/actions/runs/42237800042),
and [model submission standards](../../docs/MODEL_SUBMISSIONS.md).

## Evaluate a trained model

Install the public `codeccat==0.1.0` Runtime only in the evaluation environment, then compare against the official bundled model:

```bash
python -m codeccat_train evaluate --model custom.ccm \
  --manifest dataset/test-manifest.jsonl --split test --report comparison.json
```

This compares decoded Unicode text (primary) and exact codec labels (secondary), with per-codec results and paired improves/regressions. `--split validation` can be used for development, but **do not call it an independent holdout**. The command does not replace or update the official model.

## Optional L3 and historical reproducibility

For the optional L3 Unicode-trigram artifact, the **same offline Trainer**
supports a train-only manifest. Unlike the L2 `--residual-focus` replay,
L3 uses every eligible unfiltered input record:

```bash
codeccat-train fit-l3 --manifest dataset/train-only.jsonl \
  --output language-l3.json --report l3-report.json
```

L3 stays disabled by default. The pinned historical L2 CCM2 and L3 JSON
have both been reproduced byte-for-byte in GitHub Actions; see
[official source and reproducibility status](../../datasets/official/README.md).
The manual strict replay gate fails when required hashes, source
fingerprints or numeric identity evidence differ. This is **not** rights
clearance or authorization to replace a production model.

See [quick start](docs/GETTING_STARTED.md). The legacy `python -m codeccat_train --manifest ...` training command remains supported.

## Compact CCM2 export

The same trainer can export `CCM2` float32 weights instead of the default `CCM1` float64 weights:

```bash
codeccat-train fit --manifest dataset/manifest.jsonl --output custom-f32.ccm \
  --report train-f32.json --precision f32
```

`--precision f64` remains the default for compatibility. CCM2 stores IDF and bias as float64 while compressing the weight matrix to float32. A smaller model does not imply better accuracy; compare both models on independently held-out bytes before choosing one. This is a format option for custom models, **not** reproduction of the published official CCM2 model.
