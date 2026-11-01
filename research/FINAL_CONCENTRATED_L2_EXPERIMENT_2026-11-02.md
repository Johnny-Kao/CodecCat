# Final concentrated L2 experiments — 2026-11-02

## Executed
- Five-way matrix: https://github.com/Johnny-Kao/CodecCat/actions/runs/41362355733 — PASS (5/5).
- Same-run paired Stable23 versus binary 1–2 gram: https://github.com/Johnny-Kao/CodecCat/actions/runs/41362464702 — PASS.

## Matrix (all 3086 grouped held-out chardet fixtures; external set only 8)
- Stable23 baseline: historical benchmark 2828 (same-run paired), candidate 1–2 gram 2868, balanced 1–2 gram 2853, wider 1–2 gram 2868, **binary-term-frequency 1–2 gram 2882 (93.39%)**.
- Top-3 correct decoding absent: binary 71 vs prior Stable23 105.
- External eight: all candidates 7/8, insufficient to establish source-independent generalization.

## Paired same-job audit
- Stable23 2828/3086 (91.64%); binary 1–2 gram 2882/3086 (93.39%). Net +54 (+1.75 percentage points).
- Binary-only successes 93; Stable23-only successes 39.
- Improve 31 encoding families, regress 13; 75 aggregate family wins, 21 losses.
- Losses: cp273 -3, cp850 -3, cp1252 -2, cp500 -2, cp858 -2, iso8859-14 -2 and seven additional families -1.
- Gains: mac-roman +7, utf-7 +6, hp-roman8 +6, koi8-u +5, cp1250 +5, cp037 +4, utf-8 +4.
- CPU p50 2.039 ms baseline vs 1.448 ms candidate; p95 worst fold 3.738 vs 2.556 ms.
- Training CPU totals 16.268 sec vs 7.850 sec; process peak RSS for both 295.36 MiB in same job: this is NOT inference-only memory and cannot be attributed as a RAM improvement.

## Blocking issues
- 13 encoding families regress; cannot claim full nonregression.
- Almost all evaluation examples derive from competitor-owned development corpus; 8 unseen exact-byte external examples are too few.
- Runtime portable model loader currently restricts TFIDF parameters to fixed schema with no binary-term-frequency field. A true promotion requires a new portable schema / verified save-load parity.
- Historical 2-case fluctuations in Stable23 across CI runs not fully explained; compare pinned exact artifacts and model fingerprints before release candidate freeze.
- The binary variant is a **validated exploratory winner, NOT a deployable release model**. Do not modify main, Alpha, or PyPI until the blockers are cleared or explicitly accepted.

## Decision
Stop broad hyperparameter searching now. If advancing the candidate, perform a single independent-source holdout and full package model roundtrip/end-to-end CPU/RAM correctness test. Otherwise retain Stable23 as default and complete Alpha packaging.
