# Frozen confirmation protocol — CPython CJK encoding fixtures

**Registered before scoring on 2026-11-07.** This is a new source, not the
previously inspected GTK or uchardet datasets.

- Source: `python/cpython`, revision
  `f64e51563a85667446793ad61ea521d005ceb171`, directory
  `Lib/test/cjkencodings/`.
- Every paired `NAME.txt` and `NAME-utf8.txt` present at the pinned
  revision is included, lexicographically ordered. No selection by outcome.
- `NAME.txt` is decoded with Python's canonical `NAME` codec; the paired
  `NAME-utf8.txt` is decoded as UTF-8. If reference text does not match,
  the protocol fails rather than silently filtering.
- Evaluate each *complete file* once, including the UTF-8 reference (two
  inputs per pair). Do not trim, clean, chunk or adjust based on outcomes.
- Only compare actual Unicode text, not just the predicted codec label.
  Published `codeccat==0.1.0` versus pinned `chardet==7.7.0`.
- Existing baseline predictions are compared with UTF-8-rescue thresholds
  of 1, 2 and 3 actual non-ASCII Unicode characters, with the same
  minimum size, BOM and control-char restrictions as the research script.
- Before scoring: exclude any exact full-file or 4096-byte prefix overlap
  with the pinned historical training corpus. If any is found, fail closed
  and report metadata only; then the source cannot count as independent.
- This screen cannot prove semantic independence, source license clearance,
  or coverage of cp1252/ISO-8859-15. The CPython corpus may also have been
  used by other projects or their sources. A negative screen is not a
  blanket clearance for release.
- Per-source path/name, length, hashes, expected encoding, scores,
  paired help/harm and aggregate counts may be recorded. No original
  bytes, test strings, trained model, or inferred personal data is uploaded.
- No model adjustment, new default, Stable Runtime change or PyPI action
  follows automatically from this experiment.

**Licensing:** CPython repository is distributed under PSF License
Agreement; this does not eliminate possible individual fixture provenance
questions. Keep raw fixtures in ephemeral GitHub Actions checkout only.
