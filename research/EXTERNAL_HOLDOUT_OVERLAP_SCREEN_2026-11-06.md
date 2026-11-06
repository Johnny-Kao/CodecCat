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

The near-overlap count is inclusive of exact matches; it is **not** an additional 50 windows. The remaining 17 flagged windows are candidates requiring source review, not proven duplicates. In this screen, at least 33 of the 102 uchardet evaluation windows directly reuse bytes from the historical L2 training corpus. That means the published 94/102 CodecCat result **must not** be described as performance on 102 independent, untouched examples. Scoring only non-exact windows would still not make that subset blind: the results have already been observed and the near-overlap candidates remain unresolved.

GTK had no detected raw-byte/prefix/sample-span overlaps against the pinned L2 input. This does not prove true semantic or source-level independence.

The screen stores the external input SHA256 and source paths, plus candidate counts, without raw text. It does not test semantic paraphrases or canonicalized cross-encoding Unicode matches. Source rights remain unreviewed.

**Next quality work:** prioritize previously identified decoding failures as development regressions, then acquire a new, independently sourced and preregistered confirmation set. This audit does not authorize an official model promotion or release.
