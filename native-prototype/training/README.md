# codeccat-train (offline only)

Independent Python distribution for training and exporting CodecCat native
`.ccm` models. **Not a runtime dependency** of `codeccat`. The train
environment requires NumPy and scikit-learn; production inference does not.

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

## Evaluate a trained model

Install the public `codeccat==0.1.0` Runtime only in the evaluation environment, then compare against the official bundled model:

```bash
python -m codeccat_train evaluate --model custom.ccm \
  --manifest dataset/test-manifest.jsonl --split test --report comparison.json
```

This compares decoded Unicode text (primary) and exact codec labels (secondary), with per-codec results and paired improves/regressions. `--split validation` can be used for development, but **do not call it an independent holdout**. The command does not replace or update the official model.

See [quick start](docs/GETTING_STARTED.md). The legacy `python -m codeccat_train --manifest ...` training command remains supported.

## Compact CCM2 export

The same trainer can export `CCM2` float32 weights instead of the default `CCM1` float64 weights:

```bash
codeccat-train fit --manifest dataset/manifest.jsonl --output custom-f32.ccm \\
  --report train-f32.json --precision f32
```

`--precision f64` remains the default for compatibility. CCM2 stores IDF and bias as float64 while compressing the weight matrix to float32. A smaller model does not imply better accuracy; compare both models on independently held-out bytes before choosing one. This is a format option for custom models, **not** reproduction of the published official CCM2 model.
