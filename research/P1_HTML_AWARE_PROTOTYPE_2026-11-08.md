# CodecCat P1 — Opt-in HTML-aware prototype: first engineering acceptance

**Date:** 2026-11-08  
**Scope:** isolated research only. No `codeccat` Runtime source, native model, public API, release assets or `codeccat-train` changes.  
**Implementation:** [`html_aware.py`](../prototypes/html_aware.py) (correct relative path: `research/prototypes/html_aware.py`).

## Decision

**Keep the generic CodecCat Runtime unchanged. The explicit HTML-aware architecture is viable as a prototype, but is NOT yet ready for the public API or an 0.1.2 release.**

It improves the conceptual contract by taking caller-confirmed HTML input and available HTTP `Content-Type`, preserving source-of-evidence metadata, avoiding arbitrary tag regexes, and using the WHATWG label registry. It avoids executing scripts or network access. However, its HTML tokenizer is Python's `HTMLParser`, **not a full WHATWG HTML parsing implementation**, and `webencodings` relies on CPython codecs. Exact browser Unicode mapping parity is not guaranteed, especially for Shift_JIS/CP932.

## Explicit proposed prototype contract

```python
# Research checkout only; this is NOT yet available from pip install codeccat.
from research.prototypes.html_aware import decode_html_prototype

out = decode_html_prototype(
    raw_html_bytes,
    content_type="text/html; charset=gbk",
    errors="replace",
)
print(out.encoding, out.source, out.used_replacement)
```

Data flow:

1. Require `bytes`, reject oversized inputs and invalid caller parameters.
2. Use BOM, then trusted explicit override, then HTTP `Content-Type`, then a parsed HTML `meta` tag, and finally caller-selected fallback.
3. Resolve browser aliases using `webencodings.lookup`: e.g. `gb2312` -> `gbk`; `windows-31j` -> `shift_jis`; `iso-8859-1` -> `windows-1252`.
4. Decode complete HTML source text (not DOM text), expose source and encoding, distinguish real decoder replacements from a literal U+FFFD already present in the input.
5. Never call `CodecCat.detect()`, rerank candidates, retrain a model, modify native defaults, or promote this research module to a shipping package.

`webencodings` is an explicit **research-only** dependency; it does not become part of the CodecCat production Runtime.

## Portable tests and existing public Runtime isolation

[Research prototype CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42399717508) PASS for:
- Linux Python 3.10 and 3.14; macOS Python 3.12; Windows Python 3.12.
- Real meta and HTTP-equiv, fake meta in comments / script / quoted attributes, late meta, BOM vs HTTP/override, duplicate and unknown labels, strict errors, byte-size guardrails.
- Installed `codeccat==0.1.1` preserved and unchanged; opt-in prototype remains unexported.
- A first cross-platform CI found a Python 3.10–3.12 constructor incompatibility (`HTMLParser(scripting=...)`), which was fixed rather than quietly limiting supported Python versions.

## Exact-encoding and exact-Unicode browser parity

[250-case screened original HTML browser CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42399717519) PASS.

The study reuses the 1,886 eligible documents from the pinned historical 2,675-document archive and original 3,086-source training overlap screen. It selects all 190 documents where `html5lib` and `turbohtml` disagree, plus 60 parser-agreement controls; sources stay temporary on CI. Real headless Chromium serves local HTTP `text/html` without charset, disables script execution, blocks external resources, and gives the browser encoding decision. Node's WHATWG `TextDecoder` then provides the comparison Unicode text for that selected encoding.

| Archive group | Cases | Prototype encoding matches Chromium | Prototype Unicode matches browser WHATWG decode |
|---|---:|---:|---:|
| GBK | 23 | 19 | 19 |
| Shift_JIS | 12 | 12 | **5** |
| UTF-8 | 29 | 26 | 26 |
| Windows-1251 | 146 | 146 | 146 |
| Windows-1256 | 40 | 38 | 38 |
| **Total** | **250** | **241** | **234** |

**Important:** this is **selected disagreement-enriched browser compatibility**, not true accuracy or expected production frequency. The browser had no original HTTP headers. Source folder labels are not independent Unicode ground truth.

The **9 encoding-selection differences** are a tokenizer / prescan / fallback research problem. The **7 further Unicode differences** (all from Shift_JIS) show that Python-based `webencodings` does **not** guarantee Chromium-compatible byte-to-Unicode mapping even when the chosen encoding label matches. Shipping it as browser-identical would silently produce different characters.

Incremental standalone Python prototype timing on these 250 files, Ubuntu Actions one-run measurement (not directly comparable to previously measured Runtime-only sub-ms workloads):
- p50 **9.69 ms** per call; p95 **22.12 ms**.
- Most samples used HTML meta (247); 3 used fallback. No original HTTP response headers were present.
- No accepted model or inference-quality improvement claim follows from this timing; CPU/RSS across representative new sources still needs testing.

## Release acceptance gates NOT met

1. **Decoder fidelity:** replace or validate CPython-backed mappings against WHATWG TextDecoder/encoding_rs across all official legacy encodings and a broader corpus; avoid silent Shift_JIS Unicode substitution.
2. **HTML tokenizer fidelity:** extend adversarial standards coverage, including script/raw-text, template/noscript, late meta, malformed attributes, comments, duplicates, conflicting HTTP metadata and browser locale defaults. Establish a transparent, bounded tokenizer implementation or appropriately licensed lightweight parser approach.
3. **Behavior under independent held-out sources:** verify exact Unicode and browser encoding decisions using at least two independent original HTML source families beyond the pinned archive. Report paired benefits, harms, abstentions and limits; no source-label-as-ground-truth assumption.
4. **Engineering:** ensure no mandatory parser/NumPy/sklearn dependency added to generic Runtime, keep explicit opt-in and stable `detect(bytes)` contract, benchmark p50/p95 + RSS and very large/adversarial HTML, document fallback/override security boundary.
5. **Scope:** no new version, no public import `codeccat.decode_html`, no upstream PR until a separate production review.

**Conclusion:** The architecture passes a meaningful proof of concept, not production compatibility. Keep Runtime `0.1.1` Stable, keep the prototype in research and continue only along verified browser-compatible implementations.

### References

- [Previous P1 real-browser comparison](P1_BROWSER_ARBITRATION_AND_GBK_SOURCES_2026-11-08.md)
- [WHATWG HTML encoding sniffing](https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding)
- [WHATWG Encoding](https://encoding.spec.whatwg.org/)
- [webencodings API reference](https://doc.courtbouillon.org/webencodings/stable/api_reference.html)

All downloaded HTML stays ephemeral; research artifacts contain only aggregate results, sanitized file hashes and parser labels.
