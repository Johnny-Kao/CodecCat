# CodecCat P1 — Three-track encoding research (2026-11-07)

**Scope:** Research only, published `codeccat==0.1.1` unchanged. No model weights, Runtime defaults, ABI, interface or PyPI release modified. PR #55 and [GitHub Actions run 42382058417](https://github.com/Johnny-Kao/CodecCat/actions/runs/42382058417) PASS. All original datasets were checked out temporarily; outputs contain only counts, input SHA256 and source paths, never source file bytes or decoded user text.

## Executive decision

1. **Benchmark semantics — DONE for now.** Prior [PR #54](P1_WHATWG_DUAL_ORACLE_FINDINGS_2026-11-07.md) demonstrates why Python `shift_jis` exact-Unicode scoring and web `Shift_JIS` scoring answer different questions; do not retrain based on those mismatches.
2. **HTML-aware detection — PROMISING, but `html5lib 1.1` is not enough for conformant production behavior.** Compare its binary prescan to metadata and adversarial bytes before building an opt-in resolver.
3. **GBK independent source validation — INSUFFICIENT.** New-source fixtures screened this round are too small, empty or mislabeled. One previously examined CPython case is correct; do not claim improvements or use these data to tune the model.

## Track 1 — dual-oracle benchmark ontology

From the 1,886 source-screened IUST archive documents, scoring the unchanged CodecCat output under strict Python mapping vs browser-compatible WHATWG TextDecoder shows:

| Archive label | Documents | Python strict Unicode agreement | WHATWG Unicode agreement |
|---|---:|---:|---:|
| Shift_JIS | 532 | 239 | **516** |
| GBK | 356 | 115 | **115** |
| UTF-8 | 333 | 325 | 325 |
| Windows-1251 | 219 | 214 | 214 |
| Windows-1256 | 446 | 421 | 421 |

The source labels are **not independently verified truth**. Even the browser-column scores are only directory-label agreement. For a fair regression suite, store: original byte hash, source group, encoding family, Python vs WHATWG text hashes, missing codec class, candidate recoverability and selection reason. Do not collapse the two semantics into a single “correctness” percentage.

## Track 2 — explicit HTML context and prescan

Study: 1,886 screened original HTML bytes; compare the narrow first-1024-byte regex from PR #53 with `html5lib._inputstream.HTMLBinaryInputStream` v1.1 with `useChardet=False` and windows-1252 fallback. **This is an exploratory parser implementation, not browser-equivalent conformance verification.** Original HTTP headers were not collected.

| Folder label | Documents | Parser-selected encoding family matches folder label | Different |
|---|---:|---:|---:|
| Shift_JIS | 532 | 532 | 0 |
| GBK | 356 | 333 | 23 |
| UTF-8 | 333 | 306 | 27 |
| Windows-1251 | 219 | 83 | 136 |
| Windows-1256 | 446 | 382 | 64 |

A selected label matching the archive label does **not** prove decoded text correctness or that the label should be trusted; the historical archive may have been created using the same HTML declarations. Different cases may reflect no declaration and windows-1252 fallback, missing transport headers, legacy alias normalization, or incorrect metadata. These aggregates should not be advertised as accuracy.

### Adversarial prescan checks

| Fixture | Expected from its setup | html5lib selection | Result |
|---|---|---|---|
| Genuine `<meta charset=gbk>` | GBK | GBK | PASS |
| `http-equiv=Content-Type` | GBK | GBK | PASS |
| Fake meta inside HTML comment | windows-1252 fallback | windows-1252 | PASS |
| Fake meta inside JavaScript string in `<script>` | windows-1252 fallback | **GBK** | **FAIL** |
| Meta beyond first 1,024 bytes | windows-1252 fallback | windows-1252 | PASS |
| HTTP transport UTF-8 vs meta GBK | UTF-8 | UTF-8 | PASS |
| UTF-8 BOM vs meta GBK | UTF-8 | UTF-8 | PASS |

**The script-string false positive is a stop sign.** Do not ship this exact html5lib prescan as a claim of WHATWG conformance. A future HTML-aware feature must use full HTML tokenizer-prescan rules, verify `script`, comments, attribute quoting, `http-equiv`, BOM and transport priority against independent reference implementations, and specify fallback behavior. The base `detect(bytes)` has no document type or HTTP header context: do not inject HTML-specific heuristics into it.

Official standards: [WHATWG HTML encoding sniffing](https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding), [WHATWG Encoding label mapping](https://encoding.spec.whatwg.org/). [ICU](https://unicode-org.github.io/icu/userguide/conversion/detection) explains that HTML markup distorts language statistics and offers an optional markup filter, not a universal format-agnostic replacement.

## Track 3 — external GBK sample quality and candidate selection

Sources pinned in CI: previously examined CPython `f64e51563a85667446793ad61ea521d005ceb171`, first new external `coodoing/piconv` at `3a450b6a6f2c92e9a359ac7a91afe9551d4a120c`, and an independent charset-normalizer repository negative-control candidate. Historical training-source overlap screening uses full content, first 4 KiB, and sampled block matches against 3,086 pinned historical files.

Quality requirements: original bytes, strict claimed-label decoding, **>=64 bytes**, negative historical overlap screen, and at least two Han codepoints for a new purportedly GBK document.

| Source | Original bytes | Quality finding | Scoreable? |
|---|---:|---|---|
| piconv Chinese ANSI sample | 4 | Too short to distinguish encoding reliably | No |
| piconv GBK example | 68 | Invalid strict GBK under claimed label | No |
| piconv GBK recovery file | 0 | Empty | No |
| CPython `gbk.txt` | 755 | Strict GBK, 273 Han codepoints, negative overlap screen | **Control only (previously examined)** |
| charset-normalizer Chinese UTF-8 sample | 743 | Fails historical training-source overlap screen | No |

On the CPython control, CodecCat selects `gb18030` and produces the **same Unicode as strict GBK**, so this is a family alias/output equivalence success, not proof its exact label equals GBK. **There are zero newly eligible, sufficiently informative independent GBK documents in this round.** A percentage based on these cases would be misleading.

### Research candidate, not a patch

Prior larger IUST results remain useful for *hypothesis discovery*: 241/356 GBK directory-label mismatches, 163/241 recoverable using a returned candidate's Python strict decoding against WHATWG source-label Unicode. However those are single-source proxy judgments; `GBK` absent from top 3 and abstention cases require a separate fix path.

To justify retraining or a general-purpose reranker, require:
- at least two **new source families** of original full-length GBK documents that survive byte and source overlap checks, with documented provenance and reference bytes/Unicode;
- paired source-grouped GBK, UTF-8, Western, Japanese and GB18030 checks, counting helps **and harms**;
- stable CPU, p50/p95 latency, RSS and candidate-coverage evidence, plus a maintained semantic contract;
- source-rights review prior to official model training or redistributing fitted weights.

A newly encoded synthetic sentence does not substitute for authentic archived GBK originals. A tiny 4-byte file does not establish a new-source result.

## Next decision

**No Runtime 0.1.2, no model changes, no blind HTML parser.** Finish sourcing independently auditable GBK original files and a WHATWG-compliant HTML prescan reference before assessing any prototype. Runtime 0.1.1 and the independent Trainer 0.1.0 remain stable.

### Limits

The IUST folder names, embedded meta declarations and selected source documents may share provenance; this precludes treating them as independent oracle labels. The html5lib binary prescan is not necessarily fully WHATWG conformant, HTTP headers are missing, and this study has no new power to establish production prevalence. Original bytes are never committed or uploaded.
