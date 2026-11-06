# Official dataset provenance and re-training status

The **current published CodecCat 0.1.0 model** was fitted using [chardet/test-data](https://github.com/chardet/test-data) at commit `b0c0d206913ba118ffea2f7d306883e7a4a76db7`. The frozen fitted model fingerprint/bytes are documented in the release provenance and [RIGHTS_AND_PROVENANCE.md](../../RIGHTS_AND_PROVENANCE.md).

This directory **does not mirror the training corpus**. The upstream dataset contains files from different copyright holders with different restrictions. Publication of the corpus in a public GitHub repository is not automatically permitted, and simply providing source URLs is not a grant of training or redistribution rights.

## What is and is not reproducible

- **Available:** frozen official model, exact model SHA256, pinned source commit, published Wheel regression checks, and a standalone community Trainer for user-owned labeled corpora.
- **Not yet established:** a legally shareable complete dataset manifest, an exact `codeccat-train fit` recipe reproducing the published CCM2 model, and an independent source-level Holdout suite. The current Trainer emits CCM1 and uses a simplified training profile.
- **Rule:** do not label a community retraining output as the official Stable model unless the official training recipe and checks have been reviewed and shown to reproduce it.

Next step is to audit the sources/permissions, codify the official training profile in the same Trainer (not a second training product), and verify reference-model identity or document any unavoidable differences. Do not commit raw data until permission is confirmed.

See [Trainer guide](../../native-prototype/training/docs/GETTING_STARTED.md) and [ROADMAP.md](../../ROADMAP.md).
