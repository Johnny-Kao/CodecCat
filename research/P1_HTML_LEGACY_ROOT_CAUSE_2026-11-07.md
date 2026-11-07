# P1 — Legacy HTML encoding discrepancy analysis (2026-11-07)

**Research only.** No Runtime or bundled CCM changed, no model retrained, and no new default heuristic proposed. PR #53; [metadata-only GitHub Actions run](https://github.com/Johnny-Kao/CodecCat/actions/runs/42371476566) PASS.

## Scope and reproducibility

- Frozen **published** CodecCat Runtime `0.1.1` and the original IUST-HTMLCharDet 2,675-document crawled HTML archive, pinned commit `0a7f382e2d488d46f1e12bad647f0384c4a2288f`.
- Reused the previous full-blob/first-4 KiB/multiple-byte-block screen against 3,086 historical training sources, fixed encoding-label strict decoding and deduplication. **1,886 eligible raw HTML documents**.
- Group directory codecs are **proxies for Unicode truth**. These documents are historically correlated and have not been independently reviewed. This study does not establish production accuracy.
- Only aggregates and file hashes are recorded; source HTML/text stays ephemeral on GitHub Actions.

## New findings (exact Unicode agreement with directory-label decoding)

| Cohort | Eligible | Agreed | Disagreed | In-disagreement label in top 3 | Early HTML meta matched source label |
|---|---:|---:|---:|---:|---:|
| Shift_JIS | 532 | 239 | 293 | 293 | 532 / 532 |
| GBK | 356 | 115 | 241 | 150 | 333 / 336 meta-bearing |
| Windows-1251 | 219 | 214 | 5 | 4 | 90 / 100 meta-bearing (9 unrecognized labels) |
| Windows-1256 | 446 | 421 | 25 | 16 | 385 / 393 meta-bearing (1 unrecognized) |
| UTF-8 | 333 | 325 | 8 | 7 | 306 / 306 meta-bearing |
| **Total** | **1,886** | **1,314** | **572** | — | — |

The early meta counts use a **very restricted ASCII regex on the first 1,024 bytes**, not a WHATWG-compliant prescan, HTTP header inference or semantic claim about true authorship.

### Shift_JIS: primarily Unicode mapping semantics, not wrong family

- **All 293** apparent Unicode mismatches were labeled `shift_jis` and **decoded by CodecCat exactly like Python `cp932`**; none chose a different family or abstained.
- Of all 532 Shift_JIS documents, **516 CodecCat outputs equal strict Python `cp932` decoding**. In **309/532** original byte streams, Python `cp932` and Python `shift_jis` produce different Unicode strings.
- Examples: bytes `81 60` become U+301C under Python `shift_jis` but U+FF5E under `cp932`; `81 7C` maps to U+2212 vs U+FF0D. These are **meaningful distinct Unicode scalars**, so we must not globally normalize them or declare them harmless; the right interpretation depends on the intended mapping contract.
- CPython documents its separate CJK codec tables and their compatibility choices. The WHATWG web `Shift_JIS` encoding treats labels like `windows-31j` as web aliases; it is not identical to Python's `shift_jis` codec contract in every case.
- **Conclusion:** the low *strict-Python-label agreement* figure does not demonstrate widespread Japanese language misclassification. A dual benchmark (strict Python mapping vs WHATWG browser-compatible decoding) is required **before** selecting any improvement.

### GBK: a genuine candidate for investigation

- **241/356** were Unicode-inconsistent with directory-labeled Python `gbk`; **150** had the correct family codec in the ranked candidates, **91** did not; **56** abstained.
- **114** originally selected `gb2312`, and 40 of those produce Unicode different from strict Python `gbk`. WHATWG defines `gb2312` as a GBK label on the web, which creates a similar evaluation ambiguity; **this does not explain the remaining non-Chinese predictions**.
- Among 241 disagreements, **230** had an early meta declaration matching the archive's GBK label, **3** had a conflicting early meta declaration, **8** lacked it. This supports evaluating an **explicit HTML-aware pipeline**, not rewriting generic whole-buffer detection to trust arbitrary markup.
- Frequent non-Chinese outputs include cp1256 (61), ISO-8859-15 (27), ISO-8859-4 (17), koi8-u (15); these are credible candidates for source-sensitive language/byte-structure research, rather than a Shift_JIS-specific label alias fix.

### ASCII-tag-only counterfactual — NOT a proposed algorithm

A deliberately restricted regex replaced ASCII-only tags by a single ASCII space before calling **the same frozen model**, while measuring correctness against the corresponding stripped bytes. This is not a complete HTML parser; it can corrupt syntactic information.

| Cohort | Eligible mask experiment | Baseline label agreement | Masked label agreement | Paired helps | Paired harms |
|---|---:|---:|---:|---:|---:|
| Shift_JIS | 532 | 239 | 528 | 289 | 0 |
| GBK | 355 | 115/356 | 147/355 | 46 | 13 |
| Windows-1251 | 219 | 214 | 218 | 5 | 1 |
| Windows-1256 | 442 | 421/446 | 400/442 | 5 | 23 |
| UTF-8 | 333 | 325 | 313 | 0 | 12 |

The original and masked denominator differ for GBK and Windows-1256; compare **paired helps and harms** for those groups, not raw aggregate percentage changes. Do **not** enable markup stripping globally: it demonstrably harms previously label-agreeing documents, and may break legal HTML or out-of-context byte streams.

## Literature and standards reviewed

1. Gen-itiro Kikui (COLING 1996), *Identifying the Coding System and Language of On-line Documents on the Internet*: jointly scores likely decoded language and encoding. https://aclanthology.org/C96-2110/
2. Seungbeom Kim and Jongsoo Park (Stanford CS229 project, 2007), *Automatic Detection of Character Encoding and Language*: compares SVM byte-frequency and character-based naïve Bayes with Mozilla's pipeline; historical sample performance is not a contemporary general accuracy guarantee. https://cs229.stanford.edu/proj2007/KimPark-AutomaticDetectionOfCharacterEncodingAndLanguages.pdf
3. ICU's maintained Charset Detection documentation: legal byte patterns, character frequency, language trigrams and optional markup filtering. https://unicode-org.github.io/icu/userguide/conversion/detection
4. WHATWG Encoding and HTML Standards: distinguish legacy web labels, codec mappings, BOM, metadata authority and the HTML first-1024-byte encoding prescan. https://encoding.spec.whatwg.org/ and https://html.spec.whatwg.org/multipage/parsing.html
5. CPython CJK codec mapping documentation: https://github.com/python/cpython/blob/main/Modules/cjkcodecs/README

## Recommendation and stop conditions

- **Do not retrain** based on these Shift_JIS proxy label errors; do not force `shift_jis` rather than `cp932` in the generic `detect()` result. This would change Unicode output and may make web-sourced text worse.
- **Change measurement first:** add independent dual-oracle benchmark separately scoring Python codec Unicode agreement and WHATWG `TextDecoder`-compatible output, stratified by exact original source, format, file size, HTML declaration and unsupported classes.
- **Only then prototype** an opt-in HTML-aware workflow that extracts authoritative encoding metadata in standard order (BOM/transport/meta) and compares a parsed text-only statistical sample, while preserving strict Unicode output and honoring format-specific policy. Avoid global regex masking.
- **Keep GBK candidate ranking as the only clear next model-research target**, gated by additional independent sources and paired help/harm across Western legacy, UTF-8 and Japanese; keep the frozen default Runtime/model unchanged.
- **Stop** if independent source tests cannot reproduce measurable benefits without real regressions, or if source rights do not support a proposed new official model. Faster/safer current 0.1.1 is already useful.

The report and CI are research artifacts, not a request for users to enable any new behavior.
