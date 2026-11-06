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

Result status and run URL are recorded only after actual Actions execution.
Official release approval still requires independent new-source quality
confirmation and a separate legal/provenance review.
