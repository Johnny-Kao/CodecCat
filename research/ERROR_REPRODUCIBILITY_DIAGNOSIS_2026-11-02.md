# CodecCat full-corpus error diagnosis — 2026-11-02

Source run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41321213276
Reproducibility run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41354589927

## Findings
Three independently started Python processes with PYTHONHASHSEED 0, 1 and 17 produced 2828/3086 correct predictions each, and **identical per-sample predictions** (same SHA256 fingerprint; changed_count=0). This rules out Python hash seed variation in this fixed environment, **not all possible cross-run numerical/environment variation**. Earlier full mixed-source runs showed 2835 versus 2837 CodecCat correct with the same nominal dataset. Difference remains unexplained until exact source commit, package versions and individual prediction hashes are compared.

Full mixed-source benchmark latest reproducible reference: 3094 records total; CodecCat 2835, chardet 3013, charset-normalizer 2588. A prior attribution run yielded CodecCat 2837 and chardet 3013, net gap 176; another yielded net gap 178. Never combine attribution counts across executions.

Highest priority confusions for fixed grouped-fold CodecCat:
- cp500 -> cp037 (12)
- cp1250 -> iso8859-2 (8)
- iso8859-16 -> iso8859-2 (6)
- iso8859-2 -> cp1250 (5)
- iso8859-15 -> hp-roman8 (5)
- hp-roman8 -> cp1252 (5)
- cp775 -> iso8859-13 (4)

Hypothesis: most regression concentrated in overlapping single-byte codepages, where validity gate alone cannot establish correct encoding and L2 byte n-grams may be weak. The hypothesis does **not** justify brittle filename/encoding-specific thresholds.

## Next gates
1. Record pinned full commit/dependencies/model fingerprints per run; compare 2835 vs 2837 individual SHA predictions to explain historic delta.
2. Evaluate paired top-3 truth retention and score margin for error groups; determine ranking deficiency vs intrinsic ambiguity.
3. Test one small, generalizable method only after reasoning and verify on cross-source held-out data; no blind release claim.
4. Maintain frozen baseline and no changes to main, PyPI or alpha.
