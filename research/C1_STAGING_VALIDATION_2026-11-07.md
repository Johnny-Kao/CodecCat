# Staging C1 opt-in diagnostic — final validation record (2026-11-07)

**State:** Draft PR #47. No model, default detect/decode behavior, 0.1.0 PyPI version or established public Detection field changed.

## Verified architecture and platforms

- Source distribution & Python 3.10–3.14 CI: https://github.com/Johnny-Kao/CodecCat/actions/runs/42342895134 (PASS before the final C1 regex performance-only change).
- Initial six-platform full-wheel matrix: https://github.com/Johnny-Kao/CodecCat/actions/runs/42342895140 (PASS; Linux x64/ARM64, macOS x64/ARM64, Windows x64/ARM64).
- Latest post-optimization Linux source/build/real HTML validation: https://github.com/Johnny-Kao/CodecCat/actions/runs/42343587587 (PASS).
- The last optimization changes only the internal C1 character presence scan from a per-character Python loop to an equivalent compiled regular-expression search. It does not replace the selected encoding or decoded text.

## Historical real HTML test (not engineered recoding)

Pre-registered protocol: [C1_REAL_HTML_PROTOCOL_2026-11-07.md](C1_REAL_HTML_PROTOCOL_2026-11-07.md). Source archive from external IUST-HTMLCharDet historical web crawler, pinned 2015-era material, 2,675 files / 199 MB uncompressed. After structural checks, duplicate-file removal, strict decoding under the source's folder label, and full/prefix/sampled-block comparison against all 3,086 historical training sources, **1,886 original HTML byte files** were eligible. This is archive-label agreement, **not independently validated Unicode ground truth**.

| Folder label | Eligible | CodecCat Unicode agrees with label | Warning: label agreeing | Warning: label disagreeing |
|---|---:|---:|---:|---:|
| UTF-8 | 333 | 325 | 0 | 0 |
| Windows-1251 | 219 | 214 | 0 | 1 |
| Windows-1256 | 446 | 421 | 0 | 6 |
| Shift_JIS | 532 | 239 | 0 | 0 |
| GBK | 356 | 115 | 0 | 10 |
| **TOTAL** | **1,886** | **1,314** | **0** | **17** |

Observed 17 warnings were all on cases differing from the archive label; **555 out of 572 label-disagreeing inputs were not warned**. This narrow C1 warning is not a general charset accuracy detector. Real HTML may contain mislabeled encodings or browser-specific fallback semantics; do not market these proxy numbers as population true/false-positive rates. No source HTML was exported to the repository or artifacts.

## Incremental warning-call cost (Linux CI, single-run measurements)

| Folder label | Before regex, p50 μs | After regex, p50 μs | After regex, p95 μs |
|---|---:|---:|---:|
| UTF-8 | 1859.93 | 180.24 | 352.32 |
| Shift_JIS | 1692.09 | 147.72 | 242.13 |
| Windows-1251 | 5696.22 | 470.89 | 690.28 |
| Windows-1256 | 3352.49 | 284.18 | 838.91 |
| GBK | 2040.43 | 179.15 | 985.12 |

Diagnostic-only timings exclude native detection and remain variable by input length and CI load. The rule and outputs stayed identical in both runs.

## Remaining release gate

The final optimized commit must complete the six-platform matrix, and the new opt-in API is still not a substitute for reviewing legitimate C1 characters, trusted-charset inputs, and additional production evidence. Keep PR #47 Draft until that review; do not automatically merge or publish.
