# Uchardet cross-source fixed-policy validation — 2026-11-03

## Experiment
GitHub Actions: https://github.com/Johnny-Kao/CodecCat/actions/runs/41502984500

Source: BYVoid/uchardet commit `4e685757780cb3c652fc6c9ec759f62888969ec9`; corpus in `test/`.
Binary TF SGD alpha=0.0003 trained on the same pinned chardet/test-data group corpus. LanguageCoherence trained on training source, with thresholds fixed to max_l2_margin=.5 and min_language_advantage=.15 from earlier development study. No parameter updates.

## Results
- 55 fixture paths found; 3 unsupported codec filename labels; 33 excluded due to full-byte or inference-prefix SHA256 collision; **19 retained**.
- Binary TF: **19/19** decoding-text-equivalence matches filename-derived expected encodings.
- Binary TF + language reranker: **19/19**, **0 changes**, no additional successes.
- Median base CPU 1.198 ms; median language-layer call overhead .00754 ms across 19 mostly no-op cases. Does NOT measure eligible language scoring latency.
- Combined with prior charset-normalizer source filtered 8 examples (7/8), there are **27 retained observations across two third-party repositories and 26/27 correct**, but this is not a representative, independently sampled test dataset; it may contain similar text/provenance. No confidence claim justified.

## Important limits
- Filename labels are not independently validated.
- Full/prefix hashing prevents exact byte overlap, but not same-language content rewritten or near-duplicates.
- Positive results in a very small easy fixture corpus cannot establish accuracy approaching chardet.
- Language layer not proven to help outside the 3-fold development dataset.
- Formal `v0.1.0a2` and main remain unchanged.

## Decision
Keep fixed Binary TF as research best; do not ship the language reranker. Future validation should use large true independent multilingual encoding-labeled text with provenance controls, and negative binary data for abstention. Avoid retuning to this holdout.
