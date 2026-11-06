# Official training data: provenance and reproducibility status

**Status: source inventory, not a redistributable corpus and not a reproducible
Stable release recipe.** This document does not authorize mirroring third-party
text into this repository. The bundled `codeccat==0.1.0` model is frozen.

## What is known

The current Stable model was trained using text samples from
[`chardet/test-data`](https://github.com/chardet/test-data), pinned in the
earlier model-source audit to commit
`b0c0d206913ba118ffea2f7d306883e7a4a76db7`.

This upstream dataset aggregates material from multiple sources, including
CulturaX, Common Crawl-derived corpora and other projects' test samples.
Its repository explicitly notes that copyright remains with the original
publishers. Neither the `chardet` source-code license nor a dataset-level
metadata license grants a blanket license to republish every source document.

The model training process extracts short byte n-gram statistics from at most
the first 4 KiB of sampled input; the published numeric model does not bundle
the original complete documents. Model redistribution rights and the rights
to mirror input documents are distinct questions. See
[`RIGHTS_AND_PROVENANCE.md`](../../../RIGHTS_AND_PROVENANCE.md).

## What is *not* yet reproducible

The source repository commit alone is insufficient to rebuild the exact
official Stable artifact. A complete reproducibility record still needs:

1. An exact list of included relative data paths with SHA256 for each input,
   plus applicable source, attribution and redistribution permissions.
2. Frozen group-aware train/validation/holdout assignments, including a
   record of material already used during development.
3. The exact feature profile, trainer source commit, dependency versions,
   random seed, preprocessing and model-export settings.
4. A rerun of the frozen recipe with model fingerprint and compatibility
   checks against the currently shipped `CCM2` model.

The experimental `codeccat-train` currently exports `CCM1` and uses its
own configurable source manifest. A successful custom-model training run
does **not** imply it reproduces the Stable `CCM2` weights.

## Community and official data policy

Community users may provide private, independently labeled bytes with a JSONL
manifest, train locally and load the resulting `.ccm` with the public Runtime.
No private training data should be uploaded to project CI.

An official model candidate may use the **same trainer**, but promotion
requires a recorded dataset version, provenance/rights review, independent
holdout results and API/platform regression tests. The maintainers decide
whether a candidate becomes official; training a model does not automatically
replace the bundled default.

Do not commit copied third-party corpora just to make official rebuilding
convenient. For third-party sources with unclear rights, maintain references
and independently acquired local inputs until redistribution is verified.
