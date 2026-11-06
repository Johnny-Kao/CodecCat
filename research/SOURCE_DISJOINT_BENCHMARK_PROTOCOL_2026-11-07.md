# Frozen external-source benchmark protocol (2026-11-07)

Objective: confirm **published CodecCat 0.1.0** on separately sourced
real files rather than reusing the 843-case training/development mix.

This protocol was committed before obtaining any outcomes from this new
Nautilus evaluation. No model tuning is permitted using these test results.

## Fixed input selection

1. GNOME/nautilus, commit `6412aa2752fa91309b4d6e25addb8ed11afd6e69`,
   all `po/*.po` tracked at the commit, sorted by filename. Use the
   complete original bytes, no slicing, normalization or result-based removal.
   Accept only files that decode as **strict UTF-8** and contain an explicit
   `charset=UTF-8` declaration in the PO header; structural rejections are
   listed before scoring. All eligible files are one source family (UTF-8).
2. CPython, commit `f64e51563a85667446793ad61ea521d005ceb171`,
   the previously preregistered 15 `Lib/test/cjkencodings/*-utf8.txt`
   and matching legacy `*.txt` pairs (30 complete files total).
   Each paired legacy text must decode to the identical Unicode string as
   its UTF-8 counterpart. CPython **has previously been examined**: it is a
   confirmation/regression source, not a newly blind family.

The reference training source is `chardet/test-data` at
`b0c0d206913ba118ffea2f7d306883e7a4a76db7`.

## Contamination checks before scoring

Verify the frozen manifest and training-data hashes, then screen full-byte
SHA256, first 4096-byte SHA256 and >=3 matching sampled 128-byte blocks
against the historical 3,086 records. **Fail the benchmark if any eligible
external input triggers a match or overlap candidate.** A negative raw-byte
screen does not prove semantic/document-level independence across changed
formats or re-encodings.

## Primary metric and controls

Compare exact decoded **Unicode text** with strict ground truth, not just
the encoding label. The detectors are the installed published
`codeccat==0.1.0`, `chardet==7.7.0`, and
`charset-normalizer==3.5.2`; pin exact versions. For competitors,
detect the codec and independently strict-decode the unchanged input bytes.
An undecodable guess or a non-result counts as incorrect. Record source
families independently, correct/total, and the total source counts.
Do not combine CJK and predominantly UTF-8 PO into a purported
production-population accuracy.

Reports include paths, byte sizes and SHA256, counts and predictions, but
**no original source text**, training corpus or model bytes. No inference
change, model promotion, PyPI action or new release follows from this test.

Important interpretation: although the Nautilus source is distinct from the
historical training repository, it is a single GNOME software project and
its predominantly UTF-8 translations cannot validate Western legacy
encoding accuracy or general production prevalence. A new source is not
automatically a statistically representative source.
