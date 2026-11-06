# Contributing and reviewing trained CodecCat models

CodecCat accepts **model research proposals**, not automatic replacement of the Stable default model. The same offline `codeccat-train` core may be used for user-owned and official datasets; **dataset provenance and release authorization** differ.

## User-owned custom models

- Train locally using the [offline training guide](../native-prototype/training/docs/GETTING_STARTED.md), then load via `CodecCat.from_files("custom.ccm")`.
- You can keep the data and model entirely private. You **do not** need to submit them to CodecCat or grant the project rights to your corpus.
- If you want feedback, share non-sensitive aggregate results and a synthetic reproducer. No upload of confidential source data is needed.

## Propose a model improvement

Open a [Model proposal Issue](https://github.com/Johnny-Kao/CodecCat/issues/new/choose) **before** opening a replacement-model PR. Include:

1. **Goal and affected cases:** encoding families, languages, file sizes and actual Unicode decoding failure.
2. **Training recipe:** `codeccat-train` version/source revision, Python/NumPy/sklearn versions, fixed seed, settings, model format and candidate SHA256.
3. **Dataset provenance:** origin, exact source revisions, licensing/permission, class distribution, Train/Validation/Test isolation. Do **not** send raw protected text or private data.
4. **Comparison against the current published Stable model:** same input bytes, correct Unicode text as the primary measure, encoding-label correctness as secondary, per-codec outcomes, paired helps/harms and remaining failures.
5. **Real independence:** explicitly identify any data used for training or previous tuning. A source-disjoint holdout is preferable; do not call previously inspected validation data a blind test.
6. **Runtime effects:** load/decoding on supported platforms, startup/steady-state memory, inference speed and model size when relevant.

Use `codeccat-train evaluate --model candidate.ccm --manifest tests.jsonl --split test --report comparison.json` as a first step. This gives **descriptive evidence**; its output alone does not prove independent source coverage.

## Acceptance rules

A proposed official-default-model change must pass an additional maintainer-controlled gate:

- Candidate CCM parses safely within limits; license and data-origin review succeeds.
- Native default API/CLI, Python version matrix, clean Wheel, model fingerprint and numeric parity checks remain valid or are deliberately versioned.
- No unexplained regression in independently collected encoding families or security-relevant cases. Any quality trade-off requires explicit release documentation.
- Build/test results are tied to exact candidate commit, source-manifest revisions and SHA256.
- **Only the maintainer** may approve inclusion in an official Stable release; a PR merge or automated CI success is not permission to publish a new default model.

For now, do not commit new binary model artifacts to the repository without an agreed issue and audit. Prefer a temporary, clearly identified research artifact or an external reproducible recipe. A new candidate model may be shared as a separate, clearly experimental release only after review.

## Official corpus limitation

The released v0.1.0 default CCM2 model came from a pinned third-party corpus whose copyright conditions are mixed. The existing standalone Trainer uses a simplified CCM1 path, **not yet a reproduction of the exact official training recipe**. See [official dataset provenance](../datasets/official/README.md) and [rights](../RIGHTS_AND_PROVENANCE.md). Until the source clearance and reproducibility gates are complete, do not claim a model can be reproduced byte-for-byte from a public recipe.
