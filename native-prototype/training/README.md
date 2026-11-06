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
