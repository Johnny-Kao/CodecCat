# codeccat-train (offline only)

Independent Python distribution for training and exporting CodecCat native
`.ccm` models. **Not a runtime dependency** of `codeccat`. The train
environment requires NumPy and scikit-learn; production inference does not.

From a checkout:

```bash
python -m pip install ./native-prototype/training
python -m codeccat_train --manifest dataset/manifest.jsonl --output custom.ccm --report training-report.json
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
