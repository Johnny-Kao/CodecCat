# CodecCat P1 — Browser arbitration and independent GBK source QC (2026-11-08)

**Research-only outcome.** No changes to the published `codeccat==0.1.1` Runtime, frozen CCM model, public API, or independently published `codeccat-train==0.1.0`. This report follows the earlier [P1 three-track audit](P1_THREE_TRACK_RESULTS_2026-11-07.md), on original external bytes where available.

## Decision

**The first worthwhile improvement is to define a separately opted-in HTML-aware decoding contract, not to globally rerank CodecCat bytes or retrain the GBK model.** Real Chromium testing confirmed that two commonly available Python HTML implementations differ substantially in prescan behavior. A modern parser is a better reference but is not automatically browser equivalent.

## Experiment A — real Chromium with handwritten adversarial HTML

[Chromium CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42395114687) PASS. Ten original byte fixtures were delivered by a local HTTP server, with controlled `Content-Type`, and the actual `document.characterSet` was recorded after parsing.

| Case | Chromium | html5lib 1.1 binary stream | turbohtml 1.15.4 parse | Interpretation |
|---|---|---|---|---|
| Genuine GBK meta | GBK | GBK | GBK | All agree |
| HTTP-equiv GBK | GBK | GBK | GBK | All agree |
| Fake meta inside comment | Windows-1252 | Windows-1252 | Windows-1252 | All agree |
| **Fake meta string inside script** | **Windows-1252** | GBK | GBK | **Both differ from browser** |
| Fake meta in quoted attribute | Windows-1252 | Windows-1252 | Windows-1252 | All agree |
| Actual meta after first 1,024 bytes | GBK | Windows-1252 | GBK | Full parse can revise early prescan |
| HTTP UTF-8 vs meta GBK | UTF-8 | UTF-8 | UTF-8 | Transport wins |
| UTF-8 BOM vs meta GBK | UTF-8 | UTF-8 | UTF-8 | BOM wins |
| Two real meta declarations | GBK | GBK | GBK | First valid real declaration here |
| Quoted `>` inside meta attribute | GBK | GBK | GBK | All agree |

The contrived script example is **not** evidence about general text detectors: it isolates HTML tokenizer/prescan behavior. Do not generalize from ten cases to normative conformance; actual Chromium is a concrete target, not the sole possible HTML user agent.

## Experiment B — browser arbitration of original historical HTML

[Screened source/real browser CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42395637726) PASS.

- Pinned 2,675 original crawled IUST files; 1,886 eligible after duplicate, strict archive label and finite historical training-overlap screen against 3,086 source files.
- Compare the **encoding chosen** by `HTMLBinaryInputStream` and `turbohtml.parse()` on those **original bytes**. Select **all 190** documents where the two parser implementations choose different families, plus **12 parser-agreement controls from each of five archive families** (60 controls).
- Send **250** original documents to headless Chromium through a temporary local server that sends `Content-Type: text/html` with **no charset**. Disable JavaScript, block external network requests and service workers, and read `document.characterSet`.
- **Only source hashes, group-level counts, selected encoding labels and aggregate results were uploaded. Raw HTML was not uploaded or committed.**

| Source archive group | Cases | Parser disagreements represented | Browser agrees with html5lib | Browser agrees with turbohtml | Browser agrees with archive folder label |
|---|---:|---:|---:|---:|---:|
| GBK | 23 | 11 | 12 | 19 | 23 |
| Shift_JIS | 12 | 0 | 12 | 12 | 12 |
| UTF-8 | 29 | 17 | 12 | 29 | 29 |
| Windows-1251 | 146 | 134 | 12 | 146 | 145 |
| Windows-1256 | 40 | 28 | 12 | 38 | 40 |
| **Total** | **250** | **190** | **60** | **244** | **249** |

**Conditioned on parser disagreement alone (190 documents): Chromium selects the turbohtml interpretation in 184, matches neither in 6, and matches html5lib alone in 0.** The 60 parser-agreement controls were compatible with both. This is a selection-enriched compatibility experiment; **do not claim 244/250 is a population accuracy score**. The archive folder labels may simply reproduce the same original metadata as the browser.

The apparent parser agreement counts for Windows-1251/1256 require alias normalization (`cp1251` ↔ `windows-1251`, `cp1256` ↔ `windows-1256`). A first browser-summary version incorrectly treated these names as different families; final CI [42395637726](https://github.com/Johnny-Kao/CodecCat/actions/runs/42395637726) fixes this reporting error.

### What the comparisons actually justify

- **A plain `<meta>` regex or `html5lib` binary input stream cannot be assumed equivalent to full browser parsing.** The former fails comments/attributes; the latter misses the post-1,024-byte parser restart in our fixture.
- The turbohtml full parse matches Chromium on most disputed archive documents, but **the script-string false-meta test still produces a documented disagreement**. Do not directly adopt any library, parser API or heuristic as if perfectly conformant.
- Treat **format context** as an explicit interface requirement. Bytes alone do not encode HTTP headers, document type, original browser locale or user-forced encoding. `CodecCat.detect(bytes)` must not silently start honoring HTML meta tags.

## Experiment C — original external GBK source candidates

[Original-byte screening CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42394929531) PASS. Five GBK-keyword HTML files were downloaded at fixed commits from four independent GitHub projects (CatPapers, KitJs, netkiller, coolshell). Full/prefix/sampled-block historical training overlap checks, strict GBK validation, minimum 256 bytes and at least 10 Han characters were required before any GBK agreement estimate.

| Source | Size | Screening outcome |
|---|---:|---|
| CatPapers `data/header.html` | 4,430 | Strict GBK-decodable but **0 Han characters** |
| CatPapers `cat_papers.html` | 82,031 | Strict GBK-decodable but **0 Han characters** |
| KitJs `urlencode.html` | 6,098 | Invalid strict GBK, actual early meta indicated UTF-8 |
| netkiller `architect/h5.html` | 59,886 | Invalid strict GBK, early meta UTF-8 |
| coolshell `articles/4261.html` | 102,782 | Invalid strict GBK, early meta UTF-8 |

**Zero newly scored informative original GBK Chinese documents** in this round. File names and code-search snippets mentioning `gbk` are not evidence of actual source-text encoding. The previously examined CPython sample was correct but is not independent new evidence. The older IUST source shows a candidate-ranking hypothesis, not a cross-source generalization.

## Recommended engineering gate

1. Keep generic Runtime `detect(bytes)`, `decode(bytes)`, BOM rules, default model and L3 semantics unchanged. No model retraining or version bump.
2. For a possible *optional* HTML path, first define an explicit `HTML bytes + HTTP Content-Type + optional user override` contract. Validate a standard-authoritative browser edge-case matrix, including malformed tags, comments, script, BOM, transport, late meta, and cross-origin-safe local HTTP fixtures. Only then prototype and performance-test a small parser component. Do not add an unconditional heavy parser dependency to lightweight CodecCat Runtime.
3. The GBK model ranking work should remain blocked until **at least two new independently sourced, sufficiently long, correctly evidenced original GBK Chinese document families** pass disjointness/provenance checks. Then use grouped paired helps/harms against original GBK, UTF-8, Shift_JIS/CP932 and Western files with actual Unicode reference values.
4. Stop if the new HTML interface cannot demonstrate realistic browser compatibility, or if any global rerank harms existing correct outputs. No release is justified by this study alone.

## References

- [WHATWG HTML Standard: encoding determination](https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding)
- [WHATWG Encoding Standard](https://encoding.spec.whatwg.org/)
- [turbohtml encoding documentation](https://turbohtml.readthedocs.io/en/stable/how-to/encoding.html)
- [Prior CodecCat P1 dual-oracle](P1_WHATWG_DUAL_ORACLE_FINDINGS_2026-11-07.md)

**Limits:** The browser and parser samples were selected based on disagreement, historical crawled folder names are proxy labels, the local server lacks the original HTTP headers, and a fixed Chromium build is not a certification of conformance across browsers or every WHATWG corner case.
