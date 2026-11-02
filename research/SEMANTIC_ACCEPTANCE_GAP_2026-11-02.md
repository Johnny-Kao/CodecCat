# Binary false acceptance and steady-state audit — 2026-11-02

## Evidence
- GitHub Actions run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41384106533 — PASS.
- Training: same grouped 3-fold development corpus, held-out text fold zero 400 examples.
- Synthetic negative diagnostics: 98 documents from random bytes, NUL/alternating bytes and file-format signatures with randomized payloads (PNG/JPEG/ZIP/GZIP/ELF/PDF/WASM).
- Stable23: 400/400 text accepted, 98/98 negative accepted without ambiguity. 1,950 repeat calls consistent; 5.505 seconds CPU including repeated calls.
- Binary12 alpha 0.0003: 400/400 text accepted, 98/98 negative accepted without ambiguity. 1,950 repeat calls consistent; 4.180 seconds CPU.
- Whole process peak RSS 271.35 MiB for both in one process. This **cannot** measure incremental RAM of each runtime.

## Engineering diagnosis
The code's strict codec-validity condition is necessary for valid decoding but not sufficient evidence that an arbitrary byte blob is text. Permissive single-byte codec predictions cause the detector to return a decoded string for random bytes and container prefixes. This is not an exception stability failure but a *semantic acceptance policy* gap.

## Design decision
1. **Keep the released public `decode` and `detect` behavior frozen.** An uncalibrated binary veto would introduce false rejection of legitimate UTF-16, UTF-32 and uncommon EBCDIC/control-heavy text.
2. Build a separate nontext-control evaluation cohort from authentic files, with verified media type/source and collision/overlap auditing. Synthetic signatures are diagnostic controls only.
3. Introduce an explicit decision layer with observables `accepted / ambiguous / abstain`, and a reproducible abstain reason. Avoid equating a forced codec label or strict success with calibrated confidence.
4. Evaluate a **candidate policy offline**, measuring false positive rate on binary data *and* lost authentic text coverage, by encoding family, with no training on the final holdout.
5. Preserve monotonic runtime stability gates for repeated calls, BOM, truncated 4096-byte prefixes and bounded memory before any release promotion.

## Limits
The current test does not prove arbitrary-data correctness, nor does it validate millions of calls, cross-platform behavior, independent memory usage, or real-world binary false positive rates. No production heuristic or model was altered.
