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

## Preparing an immutable source input manifest

The version-pinned original corpus can be inventoried without storing any
training text in CodecCat. `prepare_manifest.py` replays the historical
directory parsing, strict-decoding filter, 4096-byte prefix deduplication,
source-group normalization and sample order.

Run the [metadata-only GitHub Actions audit](../../.github/workflows/official-source-manifest.yml)
(or run the script in an isolated research environment with an existing
checkout at the pinned commit):

```bash
python datasets/official/prepare_manifest.py --corpus /path/to/chardet-test-data
```

The tool verifies Git HEAD, rejects modified tracked corpus files and
unexpected untracked corpus input files, then writes
`codeccat-replay-manifest.jsonl` **inside the corpus root**. Its entries
contain paths, codec labels, group IDs, SHA256 of the complete bytes,
SHA256 of the first 4096 bytes, and the historical source-fold ID.
**No original text bytes are copied into the manifest.**

The independent Trainer verifies optional `sha256` and `prefix_sha256`
fields when reading each sample. The generated manifest is *train-only* by
design because the old release builder fitted the complete eligible corpus.
Keep the corpus checkout and manifest together; do not commit or upload the
third-party raw corpus.

Use the resulting manifest with `codeccat-train fit --train-only
--residual-focus --precision f32` for a **research candidate**, never as an
automatic Stable replacement. The L3 language model is still trained by
a separate historical step, and matching official CCM2 file hashes has not
yet been demonstrated.

## Numeric comparison against the frozen reference

The manual [historical Stable23 replay workflow](../../.github/workflows/official-stable23-replay.yml)
reconstructs the **L2** model from the pinned source checkout using the
historical NumPy/SciPy/scikit-learn/joblib versions and the shared Trainer's
`--train-only --residual-focus --precision f32` options. It checks the
release reference archive's SHA256 before comparing:

- the exact candidate and released CCM2 file SHA256;
- the historical order-sensitive training fingerprint;
- the original/selected sample totals and per-codec removals.

Only aggregate JSON reports and hashed manifest metadata are retained as
Actions artifacts; the candidate model and raw source corpus are not
uploaded. A job may **PASS as a diagnostic** even when model hashes differ:
the report's `byte_identical_model` and
`training_inputs_match_reference` fields determine those separate results.

This workflow does not recreate the optional L3 language model, establish
dataset redistribution rights, or authorize a replacement Stable release.

## Optional L3 reconstruction

L3 is a separate Unicode-trigram statistical model. The historical builder
trained it on the **entire, unfiltered** 3,086-example corpus, unlike the
residual-focused L2 model. L3 is opt-in and remains disabled by default.

The independent Trainer now provides `codeccat-train fit-l3`, which uses
the same manifest validation, source hashes and grouping but does **not**
apply residual sample filtering. Example on a locally held, train-only
source manifest:

```bash
codeccat-train fit-l3 --manifest /path/to/pinned/manifest.jsonl \
  --output /path/to/language-l3.json \
  --report /path/to/l3-training-report.json
```

The [manual official replay](../../.github/workflows/official-stable23-replay.yml)
fits the optional L3 twice on the same Runner, then checks its SHA256,
training fingerprint, trigram counts and serialized bytes against the
immutable release reference. Only aggregate JSON diagnostics are uploaded,
not the model with individual trigrams, candidate CCM2 or raw corpus.

L3 parity alone neither approves enabling L3 inference nor establishes data
rights. A production default-model change still requires a separate,
source-disjoint release acceptance gate.
