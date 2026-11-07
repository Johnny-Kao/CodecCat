# P1 — Verification of the 16 HTML-aware prototype / Chromium mismatches

**Date:** 2026-11-08  
**Status:** independently rerun on pinned original sources; research only, no shipped Runtime/model change.  
**Reproducibility:** [original baseline](https://github.com/Johnny-Kao/CodecCat/actions/runs/42400136589), [16-case per-file diagnosis](https://github.com/Johnny-Kao/CodecCat/actions/runs/42400956220), [browser causal control and synthetic probes](https://github.com/Johnny-Kao/CodecCat/actions/runs/42401496731).

## The counted differences are real, but not one root cause

The prior 250-original-HTML test selected all 190 documents on which `html5lib` and `turbohtml` chose a different encoding, plus 60 agreement controls, screened for exact/partial historical training-source overlap. The published `codeccat==0.1.1` default Runtime was unchanged; `decode_html_prototype` remains confined to `research/prototypes`. The browser served source bytes as `text/html` with **no HTTP charset**, JavaScript off and all off-host requests blocked. Browser-selected charset was measured using `document.characterSet`; exact Unicode was compared against Node WHATWG `TextDecoder(browserCharset)`, not adjudicated author-intended Unicode.

Baseline on this **disagreement-enriched, non-random** sample:

- 241/250 prototype-selected encoding families matched Chromium.
- 234/250 full Unicode outputs matched the browser-selected WHATWG decoder.
- Therefore **9 encoding-selection misses**, plus **7 Unicode mapping misses despite matching labels**.

### Seven Japanese Unicode mapping mismatches: confirmed

All 7 were labeled `Shift_JIS` by both prototype and Chromium, but the Python-backed `webencodings` text did not match WHATWG Unicode. Reinterpreting **the exact original bytes** under Python `cp932`, without any change to the encoding label or candidate selection, matches the browser-reference Unicode hash in **all 7/7 cases**.

For the entire 12-file Shift_JIS subset, `cp932`-decoded output matches browser WHATWG Unicode in **12/12**; the existing research prototype only matches **5/12**. This is a **read-only counterfactual**. It supports a separate HTML-specific browser mapping adapter, **not** globally changing generic `codeccat.detect` output or asserting `cp932` is identical to WHATWG across every legal byte sequence.

### Nine encoding-selection differences: three different observations

| Folder | Misses | Prototype | Chromium | Source metadata |
|---|---:|---|---|---|
| GBK | 4 | UTF-8 from HTML meta | GBK | Early UTF-8 and later GB2312 declarations; conflicting tags are **inside BODY**; first ASCII `charset` marker at byte offset 16,938; original bytes **not strictly valid UTF-8** |
| Windows-1256 | 2 | UTF-8 from HTML meta | Windows-1256 | UTF-8 and Windows-1256 meta inside BODY/DIV/TD; first marker at offset 1,776; original bytes **not strictly valid UTF-8** |
| UTF-8 | 3 | Windows-1252 fallback | UTF-8 | No `<meta>` markers or parsed charset declarations; original bytes **are strictly valid UTF-8** |

The `HTMLParser` research prototype accepts late or body-based meta labels, whereas real browser selection depends on document parsing, metadata position and possible fallback or content-dependent behavior. Mere declaration order does not establish why Chromium made its decision.

### Controlled browser ablation: correction to the earlier hypothesis

On the six GBK/Windows-1256 conflict cases, run actual Chromium under four conditions each:

- original source;
- change all discovered UTF-8 charset labels to invalid labels;
- change all discovered GBK/GB2312/Windows-1256 charset labels to invalid labels;
- invalidate **both groups**.

For **all six** originals, browser charset stayed GBK (4 files) / Windows-1256 (2 files) in **all four variants**. The controlled edits affected 6 `charset=` values per GBK document and 2 values per Windows-1256 document; source bytes never left the private test runner.

**Therefore the claim 'Chromium selected the last meta tag' is not supported and should not be repeated.** The browser result persists after invalidating those conflicting declarations. A byte/statistical/other browser fallback may have selected it; this mechanism remains **unconfirmed**. It would be incorrect to replace 'first meta wins' with 'last meta wins' as a production fix.

### Controlled browser fallback probes

Actual Chromium on loopback HTML responses without charset:

| Synthetic input | Browser selected charset |
|---|---|
| ASCII-only page, no meta | Windows-1252 |
| UTF-8 Chinese page, no meta | Windows-1252 |
| UTF-8 Japanese page, no meta | Windows-1252 |
| Invalid UTF-8 containing CP1252 accented letters, no meta | Windows-1252 |
| Fake UTF-8 meta inside script, then valid GBK meta | GBK |
| Two genuine meta declarations UTF-8 then GBK in HEAD | UTF-8 |
| Two genuine meta declarations UTF-8 then GBK in BODY | UTF-8 |

The three large archived UTF-8 pages lacking meta were selected as UTF-8 by the browser, but the synthetic UTF-8 Chinese/Japanese controls were NOT. **Strict UTF-8 validity is not sufficient to reproduce Chromium's fallback** and cannot justify forcing UTF-8 in generic or HTML-aware CodecCat. Browser locale and implementation-specific heuristics remain a possible factor; this dataset does not identify them.

## Implementation decision

**Do not publish `codeccat 0.1.2`, do not change generic `detect(bytes)`, and do not insert source-label or last-meta heuristics.**

1. Shift_JIS **mapping**: a research-only, opt-in adapter using the CP932 Unicode map is strongly supported for these seven cases. First verify the full WHATWG Shift_JIS mapping (including exceptions and invalid byte replacement) across additional inputs and compare with `encoding_rs` or a genuinely normative decoder before considering a shipping component.
2. Six conflict files: inspect HTML tag/tokenizer context and Chromium's actual fallback detector separately. The controlled ablation shows they are **not fixed by choosing the last meta**, and is a negative result preventing a misleading patch.
3. Three no-meta UTF-8: do not implement 'UTF-8 if strictly valid'. Test independent original sources with browser locale and transport metadata controlled, report paired helps/harms.
4. Preserve `HTMLParser`/CPython mapping compatibility warnings, max-byte guardrails, and provenance metadata; no public API contract changes or new required dependencies.

### Limits

This diagnosis uses historically correlated archive labels and a disagreement-enriched 250-file subset. Chrome browser output is operational evidence, not proof of author intent or cross-browser universal conformance. UTF-8/GBK charset ablation is a controlled modification of a few ASCII tokens, not guaranteed to be the only encoding-related hint in each page. Only SHA256 hashes, source group names, short encoding labels and aggregate counts were logged. No original HTML was uploaded, and no shell/local-mac tests or upstream modifications were used.
