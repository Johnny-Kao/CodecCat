# Enca external cross-source stress audit — 2026-11-03

Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41506455826

## Frozen test design
- Enca source: Project-OSS-Revival/enca, commit e6a62fbfa049c2c078963a042da89310a6a96e6e.
- Chardet test-data training source: b0c0d206913ba118ffea2f7d306883e7a4a76db7.
- Binary TF 1-2gram alpha .0003 and independent Unicode L3 fixed margin <= .5, advantage > .15. No tuning on Enca.
- 199 files surveyed in test/, 112 files with recognized encoding label, 27 exact/full-or-4096-prefix overlaps excluded, 4 failed truth decode, 81 evaluated.
- Base correctly decoded text 45/81 (55.56%). Fixed L3 45/81, only 1 changed selection, no net improvement. Median base CPU .553 ms, extra L3 call .00556 ms on evaluated cases (often no-op).
- Important: Enca encodings are parsed from filename suffix, and supported codec mappings intentionally exclude nonstandard TeX, MIME QP, special charset names. These are proxy labels, not independently audited ground truth.
- Several Enca test files are conversion variants of the *same source text*. Byte hash dedup removes collisions, but cannot remove text-equivalent cross-encoding content. As such N=81 is NOT 81 independent documents. A source-group split or content-level canonical Unicode dedup and errors-by-family is required before using the figure as a generalization estimate.
- Many observed failures are related charset confusions and UTF-8 misselections (e.g. cp855, ISO-8859-5, cp775, iso8859-4). This suggests coverage, family priors, data provenance and cross-codepage character distinctions should be reviewed.
- Previous external tests: charset-normalizer 7/8, uchardet 19/19; combining raw cases across unbalanced sources is misleading.

## Decision
No promotion of Binary TF or L3, no new parameter tuning, unchanged published alpha. Next experiment: group-by-canonical decoded text across different encodings and stratify by encoding family, text length, source, and whether expected decoding is recoverable within Top3. Separate true missing candidate from confusable but rankable cases. Confirm source labels before declaring true model generalization error.
