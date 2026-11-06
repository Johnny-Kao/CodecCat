# Historical train-to-external overlap screen (research, 2026-11-06)

Scope: compare the **published L2 training corpus** at pinned `chardet/test-data`
commit `b0c0d206913ba118ffea2f7d306883e7a4a76db7` with two previously
evaluated sources, GNOME/gtk and BYVoid/uchardet, both also pinned by Git commit.

The independent screen is defined by
[`datasets/official/audit_holdout_overlap.py`](../datasets/official/audit_holdout_overlap.py)
and runs in [`official-holdout-overlap-audit.yml`](../.github/workflows/official-holdout-overlap-audit.yml).
The CI downloads external source bytes ephemerally and uploads **only**
metadata (hashes, counts and candidate source filenames), not original text.

The checks cover:

- Exact full-byte equality with the original training files.
- Exact first-4096-byte equality with the fitted L2 training prefixes.
- Candidate overlap flagged by at least three shared hashed, sampled
  128-byte blocks of raw bytes. This is a **screen**, not a fuzzy-text or
  semantic-overlap proof.
- Document/window grouping is not claimed to be resolved simply from
  overlapping chunks; the GTK source transformations and uchardet prefix
  variants are not identical to every historical benchmark extraction.
- A negative result does **not** establish dataset independence. Different
  byte encodings, changed markup, normalized Unicode, same semantics in
  different representations, and unexamined documents can evade these tests.

Known observed quality limitations remain: GTK: CodecCat 296/328 vs chardet
328/328; uchardet: 94/102 vs chardet 96/102. These sets have already
been analyzed and may be used as regression diagnostics, **not** as pristine
blind tests for subsequent model modifications.

## Reproduced screening results

[Audit CI #42235598046](https://github.com/Johnny-Kao/CodecCat/actions/runs/42235598046) completed successfully, using the same source-window extraction as the historical benchmark (GTK 328, uchardet 102).

| Source | Benchmark windows | Exact full-byte overlap | Exact 4096-byte-prefix overlap | Candidate near-overlap by sampled blocks |
|---|---:|---:|---:|---:|
| GNOME/gtk | 328 | 0 | 0 | 0 |
| BYVoid/uchardet | 102 | **33** | **33** | **50** |

**Correction after cross-tabulating the actual matching windows:** the sets are *not* nested. Of the 33 exact matches and 50 sampled-span candidates, **26 are in both sets**, **7 are exact-only**, and **24 are candidate-only**. The union is **57/102 flagged windows**, leaving 45 with no overlap detected by this screen. Prior text incorrectly assumed all 33 exact matches were included in the sampled-span set. The 24 candidate-only windows need review; their shared blocks do not prove duplicate content. No-flag windows are not proven semantically independent. The original Alpha 94/102 and current Stable 95/102 scores must **not** be described as performance on 102 pristine, independent examples.

GTK had no detected raw-byte/prefix/sample-span overlaps against the pinned L2 input. This does not prove true semantic or source-level independence.

The screen stores the external input SHA256 and source paths, plus candidate counts, without raw text. It does not test semantic paraphrases or canonicalized cross-encoding Unicode matches. Source rights remain unreviewed.

**Next quality work:** prioritize previously identified decoding failures as development regressions, then acquire a new, independently sourced and preregistered confirmation set. This audit does not authorize an official model promotion or release.

## Current published Stable 0.1.0 quality attribution

[Current Runtime triage CI #42238721457](https://github.com/Johnny-Kao/CodecCat/actions/runs/42238721457) evaluated the same fixed samples with **published codeccat==0.1.0** and chardet==7.7.0 (the earlier 296/328 and 94/102 figures describe the **Alpha 0.1.0a4** comparison and must not be silently applied to the current release).

| Corpus/bucket | Windows | Stable 0.1.0 correct text | chardet 7.7.0 correct text |
|---|---:|---:|---:|
| GTK, no raw overlap detected | 328 | **324** | **328** |
| uchardet, exact train duplicates | 33 | 30 | 31 |
| uchardet, sampled-span candidate only | 24 | 21 | 22 |
| uchardet, no raw overlap detected | 45 | 44 | 43 |
| **uchardet total (contaminated source)** | **102** | **95** | **96** |

The 45 unflagged uchardet cases are not a new blind holdout. These data have already been analyzed, raw-byte similarity detection is incomplete, and the sample distribution was not independently preregistered for Stable 0.1.0. Use all these results for engineering triage only.
