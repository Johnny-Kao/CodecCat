# Official dataset provenance and re-training status

The **current published CodecCat 0.1.0 model** was fitted using [chardet/test-data](https://github.com/chardet/test-data) at commit `b0c0d206913ba118ffea2f7d306883e7a4a76db7`. The frozen fitted model fingerprint/bytes are documented in the release provenance and [RIGHTS_AND_PROVENANCE.md](../../RIGHTS_AND_PROVENANCE.md).

This directory **does not mirror the training corpus**. The upstream dataset contains files from different copyright holders with different restrictions. Publication of the corpus in a public GitHub repository is not automatically permitted, and simply providing source URLs is not a grant of training or redistribution rights.

## What is and is not reproducible

- **Available:** frozen official model, exact model SHA256, pinned source commit, published Wheel regression checks, and a standalone community Trainer for user-owned labeled corpora.
- **Not yet established:** a legally shareable complete dataset manifest, an exact `codeccat-train fit` recipe reproducing the published CCM2 model, and an independent source-level Holdout suite. The current Trainer emits CCM1 and uses a simplified training profile.
- **Rule:** do not label a community retraining output as the official Stable model unless the official training recipe and checks have been reviewed and shown to reproduce it.

Next step is to audit the sources/permissions, codify the official training profile in the same Trainer (not a second training product), and verify reference-model identity or document any unavoidable differences. Do not commit raw data until permission is confirmed.

See [Trainer guide](../../native-prototype/training/docs/GETTING_STARTED.md) and [ROADMAP.md](../../ROADMAP.md).

## Historical Stable23 training recipe (recovered from source)

The historical GitHub Actions workflow at commit `69883104b5` built the
candidate using `experiments/logic_first/build_installable_candidate.py`.
It pinned NumPy `2.5.3`, SciPy `1.18.1`, scikit-learn `1.9.1`, and
joblib `1.6.0`. That script:

1. Checked the `chardet/test-data` checkout's Git commit.
2. Called `experiments/logic_first/vnext_actual_adapter.py::corpus`:
   sorted `*/*` paths, parsed codec-labeled directories using the upstream
   dataset parser, rejected missing/invalid decodes, retained the first
   4096 bytes, deduplicated by prefix SHA256, and assigned source groups.
3. Applied `vnext_residual_study.py::residual_subset`, retaining at least
   two samples per codec while filtering at most 35% of structurally solved
   samples per codec.
4. Fitted `TrainableDetector.fit(..., feature_profile="stable23")` using
   **all selected training examples**, rather than holding out a
   validation split. Exported the L2 model as a numeric CCM2 artifact.
5. Separately fitted the optional L3 language statistics from the **full,
   unfiltered** sample collection. The standalone Trainer currently
   does not recreate this L3 artifact.

The independent `codeccat-train` tool has `--train-only` and
`--residual-focus` options to reproduce steps 3–4 on a locally supplied,
correctly ordered, codec-labeled training manifest. It still requires a
**verified manifest produced with the exact historical input-selection
rules** before an end-to-end model identity check is meaningful.

Example for a private, locally held manifest (NOT a production release
command):

```bash
codeccat-train fit --manifest /path/to/local/train-only.jsonl \
  --train-only --residual-focus --precision f32 \
  --output /path/to/candidate.ccm --report /path/to/replay-report.json
```

The independent Trainer's synthetic test suite validates mathematical and
residual-sample selection parity; a separate source-level reproduction
against the original published artifact has **not** passed or been claimed.
Historically recorded package versions alone do not prove cross-machine
bitwise determinism. Do not mirror the third-party raw corpus into this
repository or publish a retrained official model without rights review.
