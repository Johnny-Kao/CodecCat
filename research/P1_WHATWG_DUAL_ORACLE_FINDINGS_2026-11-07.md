# P1 dual-oracle result: distinguish actual ranking misses from codec-table semantics

**Date:** 2026-11-07  
**State:** research only. CodecCat Runtime `0.1.1`, default model, trainer and public API unchanged.  
**Reproduction:** [GitHub Actions run 42372382269](https://github.com/Johnny-Kao/CodecCat/actions/runs/42372382269) — PASS. 1,886 original historical crawled HTML documents, fixed-byte overlap screening against 3,086 historical source files. No source HTML/text is stored or uploaded.

## Research question

Is the unusually low reported GBK/Shift_JIS agreement caused by actual mistaken model choice, or by differing definitions of the same encoding family between strict Python decoders and the HTML/WHATWG ecosystem?

We compare the byte-identical frozen published CodecCat output with two **different operational reference interpretations** of the archive's directory labels:

1. Python `bytes.decode(label, "strict")`, with UTF-8 BOM handling aligned to the CodecCat default policy.
2. Node 22's ICU-backed WHATWG-compatible `TextDecoder(label, {fatal:true})`, with default Unicode BOM handling; it raises on malformed input instead of silently inserting replacement characters.

The archive's original directory labels are **not independently adjudicated ground truth**; therefore neither score establishes actual accuracy across deployment populations. All labels used for both reference outputs come from that same historical source. Our first-1,024-byte `<meta>` extractor is a narrow exploratory regex, not a conforming WHATWG HTML prescan. No HTTP `Content-Type` metadata is available.

## Exact paired results

| Source folder | Eligible | CodecCat equals Python directory-label text | CodecCat equals WHATWG directory-label text | Python/WHATWG reference text differs |
|---|---:|---:|---:|---:|
| Shift_JIS | 532 | 239 | **516** | **309** |
| GBK | 356 | 115 | **115** | **0** |
| UTF-8 | 333 | 325 | 325 | 0 |
| Windows-1251 | 219 | 214 | 214 | 0 |
| Windows-1256 | 446 | 421 | 421 | 0 |
| **Total** | **1,886** | **1,314** | **1,591** | **309** |

All 1,886 sources were strict-decodable under their *directory-label WHATWG decoder* in this experiment. This is unsurprising because input eligibility already required strict Python decoding under the archive labels.

### Shift_JIS: stop calling all mapping discrepancies model misclassifications

- Of the 293 previously reported Python-label mismatches, **293 become matches** when measured under the WHATWG Shift_JIS mapping. **No false-family selection** was observed for those 293 files.
- **16** documents show the reverse: CodecCat matches Python strict Shift_JIS Unicode but differs from WHATWG Unicode. Overall: **516/532** WHATWG-consistent outputs.
- The source byte streams have **309/532** cases where Python `shift_jis` and browser-compatible `Shift_JIS` interpret identical bytes as different Unicode scalars. Neither interpretation should be silently swapped because a different Unicode scalar is a real behavioral change.
- No general model retraining is justified by these 293 proxy discrepancies; they call for two explicit evaluation oracles and an upstream API decision if browser text compatibility is desired.

### GBK: the residual is a model selection / unsupported-candidate research problem

- WHATWG GBK mapping **does not remove any** of the original **241/356** directory-label Unicode mismatches.
- **163 of the 241** cases can be reproduced exactly under the WHATWG reference by applying a *Python strict decoder for one of the already returned candidates*. This proves a possible candidate-decoding path; it does **not** prove the correct candidate can be selected reliably without the source label.
- **78/241** cannot be recovered by the current returned candidates' Python strict decoding. They may need candidate coverage, ranking architecture, support-class or corpus analysis.
- The last P1 study found **56** abstentions and substantial non-Chinese predictions, including Windows-1256 and several ISO-8859 variants. These cases suggest genuine deficiencies, unlike the Japanese mapping discrepancy.
- `gb2312` is a WHATWG label alias for GBK but `bytes.decode("gb2312", "strict")` and `bytes.decode("gbk", "strict")` in Python are not a universal substitute. Do not equate equal label families with equal Unicode outcomes.

### Early HTML metadata: apparent improvement is not independent evidence

| Group | Valid early meta declarations producing the same WHATWG text as archive-label reference | Potential correction among baseline reference disagreements | New disagreements among originally matching |
|---|---:|---:|---:|
| GBK | 333 | 230 | 0 |
| Shift_JIS | 532 | 16 | 0 |
| Windows-1251 | 90 | 4 | 0 |
| Windows-1256 | 388 | 12 | 3 |
| UTF-8 | 306 | 8 | 0 |

These are **oracle-conditioned, in-sample comparisons**. A directory-labeled web crawl and the HTML `meta` label often encode the *same information*; using the latter to decode bytes and grading against the former can mechanically yield a large apparent improvement. It is not an independently verified accuracy win, and Windows-1256 already shows three counterexamples where naive meta interpretation harms label agreement.

The relevant HTML Standard describes roughly this order: BOM; explicit user override if provided; HTTP transport charset; an HTML prescan (typically first 1,024 bytes); then contextual/default fallbacks. **A generic `detect(bytes)` call has no HTTP metadata or declared document type**, so hardwiring these HTML rules into the generic API would change its contract and create avoidable false positives.

## Engineering recommendation

**Priority A — fix the benchmark ontology without touching the model.** Track at least: selected codec family, exact Unicode under Python strict codec, exact Unicode under WHATWG browser decoding, whether archive-provided source label conflicts with early meta, and reasons for abstention. Preserve raw-byte safety screening and document/source-group weighting.

**Priority B — separate opt-in HTML processing from generic unknown bytes.** A future `decode_html(raw, content_type=...)` or equivalent dedicated utility could implement the actual WHATWG HTML encoding prescan and mapping while honoring transport metadata. It must have a complete standards-based parser and explicit caller-supplied format context. Do not implement by globally removing ASCII-looking tags or blindly accepting an untrusted `<meta>` substring.

**Priority C — investigate GBK ranking only if a second independently sourced family reproduces it.** Candidate reranking should compare paired **helps and harms** on original GBK, UTF-8, Japanese and Western samples. Measure top-1, top-k, abstention, Unicode exact match and CPU/RSS. Guard against false Chinese positives on noisy non-HTML files. Do not re-train or promote a new Stable model without source-rights review.

**Stop condition:** No supported new default Runtime/model change if the net benefit does not generalize across independent sources or materially harms previously correct outputs. Stable 0.1.1 remains usable.

## References

- [WHATWG Encoding Standard](https://encoding.spec.whatwg.org/) — normative labels and byte-to-Unicode maps, including GBK/gb2312 and Shift_JIS/windows-31j.
- [WHATWG HTML Standard: determining the encoding](https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding) — BOM/transport/prescan priority and format dependence.
- [ICU Character Set Detection](https://unicode-org.github.io/icu/userguide/conversion/detection) — byte legality, language statistics, optional markup filtering and caveats.
- [Kikui, COLING 1996](https://aclanthology.org/C96-2110/) — joint encoding/language estimation.
- [Prior P1 mapping/candidate analysis](P1_HTML_LEGACY_ROOT_CAUSE_2026-11-07.md).

### Study limitations

Node TextDecoder is a useful WHATWG mapping implementation but not a complete HTML browser parser, and `fatal:true` differs from normal HTML replacement behavior. Corpus labels were not independently adjudicated, samples are correlated, the train-overlap check is a finite byte screen, HTML `meta` was extracted by a regex, and we deliberately did not implement a production HTML codec resolver or modify the model.

CI reads the pinned original bytes temporarily and only uploads aggregates, SHA256 and sanitized metadata.
