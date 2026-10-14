# H11 — RH/CJK Failure Mechanism Diagnosis

Source H7 immutable 1,560 rows, run: https://github.com/Johnny-Kao/CodecCat/actions/runs/38583902590
Status: diagnostics complete. R21 unchanged. Exploratory, not new holdout.

## Main results

RH n=917, final errors=70, raw Top-3 missing=31 (27 remain final errors). RH+CJK n=220, final errors=28, raw Top-3 missing=19 (all 19 remain wrong).

| True encoding | RH/CJK n | Final errors | Raw Top-3 missing |
|---|---:|---:|---:|
| shift-jis | 22 | 15 | 12 |
| euc-jp | 15 | 5 | 5 |
| gb18030 | 95 | 5 | 1 |
| big5 | 77 | 1 | 1 |
| euc-kr | 11 | 2 | 0 |

The 17 Japanese Top-3 misses account for 17/19 of all RH/CJK Top-3 misses. Of 28 RH/CJK final errors, charset-normalizer is correct on 27 and chardet 7 on 24.

RH+CJK missing Top-3 raw top-1 predictions: UTF-8 6, CP1251 5, CP1256 1, GB18030 4, Big5 2, EUC-JP 1. Seven missing truth encodings are still within the CJK family of raw top-1. Others are cross-family.

By crawl RH+CJK final errors: CC08 13/37, CC12 9/41, CC17 3/44, CC21 3/21, CC34 0/42, CC39 0/35. This cross-crawl skew is a concern. These crawls were all exposed by H5-H11.

## Decision

**Primary hypothesis:** distinguish byte-order and multibyte-state transition patterns for Shift-JIS and EUC-JP, particularly against Cyrillic single-byte and UTF-8 alternatives. Current baseline byte-frequency and byte-sum features intentionally collapse ordered adjacency; changing samples without retraining (H8) was harmful. New features must be paired with a *separate* newly trained research candidate, not substituted under frozen weights.

## Guardrails for next experiment
- Diagnose true decoding validity and length/truncation before training: a declared Shift-JIS or EUC-JP label is not necessarily the actual byte encoding; preserve labels and provenance separately.
- Use training data distinct from evaluation data, with domain/WARC-level splits and exact-byte conflict filtering.
- Predeclare one encoding-sequence model and one frozen R21 control, compare with chardet 7 and charset-normalizer.
- Never include CC-MAIN-2026-30. CC25 remains historical-only.
- Report errors per route × family, especially UTF/CP1251 guardrails; beneficial/harmful transitions; resource cost; per-crawl no-regression.
- Do not claim novel holdout status for previously observed sources.

This is not evidence that adding an unconditional Japanese override will work; the final error counts are sparse and heavily source-dependent. No promotion or retraining of R21 authorized.
