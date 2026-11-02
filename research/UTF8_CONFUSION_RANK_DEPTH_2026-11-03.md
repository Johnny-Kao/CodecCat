# UTF-8 confusion / candidate-rank depth audit — 2026-11-03

## Evidence
- UTF8 validity audit https://github.com/Johnny-Kao/CodecCat/actions/runs/41521263303 — PASS.
- Full candidate-rank audit https://github.com/Johnny-Kao/CodecCat/actions/runs/41521469819 — PASS.
- Same Enca 81 fixture stress cases, fixed Binary TF alpha=.0003, exact/full or 4096-byte prefix overlap exclusions.

## Diagnosis
- 36 wrong Enca cases; 28 predicted utf-8* and all 28 are **valid UTF-8 byte sequences**. UTF-8 strict validation is working; a simple additional strict-validity gate cannot reject these misleading yet valid alternate decodings.
- Correct decoded text among full legal learned-codec ranking:
  - rank 1–3: 8 cases
  - rank 4–10: 10 cases
  - rank 11 or later: 18 cases
- Correct label is included in the learned classifier classes for 28 of 36 errors. Do not equate trained label coverage with top-10 or functional candidate coverage.
- In these files, arbitrary syntax-valid UTF-8 can be the wrong semantic interpretation. Need source language/script evidence or codec-family priors that are independently validated. Do not hard-code an Enca label correction.
- The 36 files are dependent variants across 59 canonical text groups; results are stress diagnostics rather than population accuracy. Full-ranking audit checks text-equivalent validity on full bytes; inference itself gates candidate strictness on 4096-byte prefix, a boundary that should be considered for long documents.

## Next architecture gate
A separate **family proposal layer**, evaluated offline with the existing L2 weights fixed. Candidate families should be generated from independent evidence (e.g. codepage-specific byte sequence features and Unicode script coherence) while preserving true Unicode decoding and abstention. Compare on grouped development folds and unseen Enca/uchardet/charset-normalizer without retuning against external samples. Include CPU/per-family false positive cost. Do not promote until external held-out accuracy shows net gain without serious per-codec regressions.

No change to Alpha, main, or default inference.
