# CodecCat v0.1.0a4 — Native Alpha

**Character-encoding detection correctness improvement.** GitHub Release only; no PyPI upload.

- Recover UTF-8 text when the classifier favors a legacy single-byte encoding but the complete bytes provide strong UTF-8 evidence.
- Preserve BOM precedence and strict decoding; avoid overriding unsupported UTF-8 classes, short/ASCII-only text, and opt-in L3.
- Report uncertain evidence-based rescue with `ambiguous=True` and `reason="utf8_evidence"`.
- Add clean-wheel regression checks and paired comparisons on five pinned corpora plus newly frozen GLib catalogs.
- No model retraining, dependency additions, or Python API removals.

Validation:
- [Final tagged six-platform wheel matrix](https://github.com/Johnny-Kao/CodecCat/actions/runs/42150439965) — six PASS.
- [Published a3 vs a4 GLib paired gate](https://github.com/Johnny-Kao/CodecCat/actions/runs/42151536276) — PASS; identical model SHA256.
- [Post-release 843-case test](https://github.com/Johnny-Kao/CodecCat/actions/runs/42151051728): 819 correct Unicode decodes (a3: 805); previously unexamined GLib translation windows improved 227/285 to 271/285 (44 helps, 0 harms).

Limits: These are specific test sets, not estimates of general population accuracy. CodecCat still trails chardet on decoded-text correctness in the five-corpus test. See [full methodology](research/NATIVE_UTF8_FIX_2026-11-06.md).

Download the operating system and CPU architecture-specific wheel from GitHub Releases. Requires Python 3.10+; standard native inference does not require NumPy or scikit-learn.
