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

## Actual state of the historical model (2026-11-06)

**Engineering reproduction PASS.** The independent Trainer has reproduced the published L2 CCM2 and optional L3 artifacts **byte-for-byte** using the pinned historical corpus, scientific-library environment, and fixed source order. The manual research workflow fails closed on 52 required conditions. See [official reproducibility status](../datasets/official/README.md) and [strict replay CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42233278592). This proves reproduction of existing artifacts under the checked conditions, **not** improved generalization or permission to distribute third-party inputs.

**Independent accuracy evidence is limited.** The [overlap screen](../research/EXTERNAL_HOLDOUT_OVERLAP_SCREEN_2026-11-06.md) found **33/102 exact training-byte overlaps** and **50/102 sampled-block candidate overlaps**; their intersection is 26, and **57/102 have at least one overlap flag**. The 24 candidate-only matches are not proven duplicates. Treat the historical Alpha 94/102 and current Stable 0.1.0 95/102 scores as descriptive only. The 328 GTK windows had no detected raw-byte overlap, but they form a narrow, previously inspected genre and do not establish general model quality. Existing observed test cases are suitable for regression development, **not** pristine new blind validation.

**Release promotion remains blocked pending separate decisions:**

- **Data rights:** review the individual upstream fixture origins and permissions, document the basis for training/model distribution, and resolve or explicitly document material unresolved rights risks. Do not assume the source repository's public accessibility is a license.
- **Quality:** pre-register a genuinely new, non-overlapping confirmation source before inspecting results; require correct decoded Unicode as the primary metric, per-codec effects and paired help/harm against the current Stable model. Record accepted trade-offs and ensure no unexplained security or decoding regressions.
- **Platform compatibility:** Trainer synthetic CCM1/CCM2 generation and published Runtime loading passed on Linux ARM64 as well as Linux x86-64, Windows and macOS Intel. **macOS ARM64 Trainer E2E is now independently verified** on Apple Silicon [GitHub Actions #42237800042](https://github.com/Johnny-Kao/CodecCat/actions/runs/42237800042), including native CCM1/CCM2 fitting, unit tests and loading with the published Runtime.
- **Final release decision:** the maintainer reviews the exact source revision, model SHA256, test provenance and rights assessment. CI success and a merged PR never automatically publish or promote a model.

The official L3 inference behavior remains **disabled by default**; successful L3 retraining alone does not justify enabling it. See [rights and provenance](../RIGHTS_AND_PROVENANCE.md).

**Current released-runtime failure evidence:** On the fixed, already-observed GTK corpus, `codeccat==0.1.0` decodes 324/328 correctly versus chardet 328/328; on the previously contaminated uchardet corpus, 95/102 versus chardet 96/102. These are descriptive research outcomes, **not** permission to promote a new Stable model or claims of independent blind accuracy. See [current Stable triage](../research/EXTERNAL_HOLDOUT_OVERLAP_SCREEN_2026-11-06.md).
