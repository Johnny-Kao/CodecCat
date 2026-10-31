# CodecCat — Real-source accuracy execution, source gating, and UTF-8 evidence

**As of 2026-11-01 (Japan local time).**  
**Research branch:** `research/accuracy-data-gate-v2`.  
**Product status:** NO MODEL PROMOTION. `main`, existing Alpha release and PyPI remain unchanged. CPU, memory and inference speed are deliberately out of scope until correctness passes.

## What actually changed

- A fully reproducible **real-source-byte benchmark protocol** freezes original upstream files, validates codec metadata, extracts translation-only original byte windows (no charset headers), records per-file SHA-256 and locks an immutable manifest **before model scoring**.
- `src/codeccat/vnext_selective.py` (introduced previously): a Latin-family specialist is allowed to overrule only compatible Latin/Windows baseline outcomes, with no Unicode/EBCDIC or BOM override.
- **NEW:** `src/codeccat/vnext_utf8_evidence.py`, an **opt-in** `Utf8EvidenceDetector` adapter. It delegates model training and the original confidence/ranking to stable23, then optionally selects a supported UTF-8 class when: final full object; no BOM; raw >=64 bytes; at least 16 decoded non-ASCII codepoints; UTF-8 strict decoding succeeds for both inference prefix **and entire object**; no unexpected Unicode Cc control characters. It marks the chosen encoding `ambiguous=True` because byte validity cannot establish the historical codec.
- The adapter never overrides non-final streaming prefixes. Important: some 4,096-byte samples used by research harnesses are **truncated prefixes** (`final=False`), not complete documents. Early counterfactual estimates incorrectly treated those as full objects. Reconciled the actual adapter with the risk screen and added an invalid-suffix-after-valid-prefix regression.
- Contract tests: [Actions #41239001834](https://github.com/Johnny-Kao/CodecCat/actions/runs/41239001834), **11 tests PASS**. Actual-endpoint/policy equivalence: [Actions #41239003451](https://github.com/Johnny-Kao/CodecCat/actions/runs/41239003451), **SUCCESS**.

## Independent byte sources, pinned before scoring

### 1. Historical uchardet test files — source contamination gate

- Source `BYVoid/uchardet@4e685757780cb3c652fc6c9ec759f62888969ec9`, frozen [#41237703300](https://github.com/Johnny-Kao/CodecCat/actions/runs/41237703300), SHA `e23005af8b0f5cb85e6b02586de5fa5ba520cc3d2cdb3e826ec8d6267faffd98`.
- 54 source files, 52 strictly valid labeled bytes, **36 overlap in decoded text with current training data**, 2 unsupported codec labels (not silently treated as valid).
- Only **16** decontaminated examples. The original source-size gate failed as intended; a separate, explicitly **underpowered diagnostic only** [#41238022093](https://github.com/Johnny-Kao/CodecCat/actions/runs/41238022093) found: baseline **15/16**, Latin-only **14/16**, routed **15/16**, chardet7 **14/16**, charset-normalizer **16/16** correct decoded text. **No promotion or broad population inference is justified.**
- This case demonstrates why many public encoding test suites cannot be naively labeled independent: benchmark samples can overlap with the model's original training provenance.

### 2. Vim actual PO translation bytes — first substantive external domain

- Source `vim/vim@cbe1cf017d9027c8e34599333ef69d7e20e1bf8d`. Frozen [#41238184283](https://github.com/Johnny-Kao/CodecCat/actions/runs/41238184283), manifest SHA `3e20642290ef06a1f1dc7ccd19c3a1264601471aea5989965beb7c96d063c992`.
- Original source-controlled `.po` translation `msgstr` raw bytes are extracted, without the `charset=` declaration or English `msgid` content. All windows must strictly decode with the PO-declared codec. Four 2,048-byte windows per eligible catalog; some are Vim-generated alternate encoding variants. **Do not claim independent authorship per window.**
- 44 source catalogs; 42 contributed eligible byte windows; 162 frozen windows; 3 duplicated raw windows excluded; **159 scored**. Paired one-shot scoring [#41238286450](https://github.com/Johnny-Kao/CodecCat/actions/runs/41238286450):

| Detector | Correct decoded text / 159 | Exact codec label / 159 |
|---|---:|---:|
| Stable23 incumbent | 155 | 132 |
| Global Latin-only updated weights | 154 | 129 |
| Selective Latin router | 155 | 132 |
| chardet 7.7.0 | **159** | 129 |
| charset-normalizer 3.5.2 | 143 | 113 |

- Four incumbent wrong decoded-text windows: **three `hy.po` Armenian UTF-8 samples predicted as TIS-620**, one `af.po` Afrikaans `iso8859-1` sample predicted `hp-roman8`.
- The new UTF-8 structural adapter was proposed only *after these cases were inspected*. It **cannot claim blind confirmation on Vim**.
- Follow-up actual adapter regression [#41239003451](https://github.com/Johnny-Kao/CodecCat/actions/runs/41239003451): threshold16 rescues **3/159**, zero harms, incumbent 155→158. On source-document isolated chardet development folds, only **+1 net / 3,086** when `final=False` is correctly honored on truncated 4,096-byte prefixes; zero harms. The earlier +3 on development was a **counterfactual that incorrectly treated truncated bytes as final** and is superseded.

### 3. Git real translation source — NEW frozen independent confirmation

- Source `git/git@50fe5cc63f9cd6e89b87a2c9ccf96a444f4cfdb7`. The same extraction method and minimum evidence threshold were frozen **before results**, [#41239218591](https://github.com/Johnny-Kao/CodecCat/actions/runs/41239218591), manifest SHA `2b966b9a344792cd3cc9aa8df14d7bbc815f8078061f14b0cf121f0b9057a3d2`.
- 22 catalogs, 20 eligible, 79 original byte windows; 3 repeated raw windows excluded, **76 evaluated**. All 76 evaluated windows are UTF-8. Actual strict-decoded-text results, [#41239422354](https://github.com/Johnny-Kao/CodecCat/actions/runs/41239422354):

| Detector | Correct decoded text / 76 |
|---|---:|
| Stable23 | **76** |
| Latin-only global retrained model | 73 |
| Selective Latin router | **76** |
| Optional UTF-8 structural rescue | **76** |
| Combined router + UTF-8 rescue | **76** |
| chardet 7.7.0 | **76** |
| charset-normalizer 3.5.2 | **76** |

- **No regression** for optional UTF-8 rescue on a new domain, but **no positive gain** either because the incumbent was already perfect on this corpus. The Latin-only model introduced three erroneous decodes of previously correct UTF-8; the selective router preserved the baseline. Do not describe this as a proof of superiority or as evidence that non-UTF-8 false-positive risk is eliminated.
- Git windows are correlated within each of 20 source catalogs. Real source bytes can be build-generated `.po` text, not a representative distribution of all deployed encodings.

## Decisions

1. **Keep stable23 as the published default.** No accuracy claim against chardet 7.7.0; chardet scored higher on Vim and matched on Git. Do not publish to PyPI or change release assets.
2. **Retain the UTF-8 structural rescue as opt-in experimental code only.** The release model may not silently change decoding decisions; the guard must prove itself on another previously unseen diverse legacy-encoding source before a proposal to expose it as a public API.
3. **Retain the Latin specialist as an experimental conditional router only.** Unconditionally applying global Latin contrast trained weights introduces real UTF-8 regressions on unseen Git; routing avoids them, but adds operational model-state complexity.
4. **Prefer more diverse genuinely independent source-controlled files** with non-UTF-8 byte content and confirmed charset metadata. Next mandatory promotion gate must group by originating document and language, detect text overlap against training and previous public test fixtures, compare exact decoded text/label/risk coverage, and pin exact runner, code, model and dataset versions. Already inspected Git/Vim/CC0/uchardet files are consumed for tuning decisions; do not relabel them fresh.

## Source links

- [UTF-8 evidence source code](../src/codeccat/vnext_utf8_evidence.py)
- [UTF-8 safety tests](../tests/test_vnext_utf8_evidence.py)
- [UTF-8 actual semantic risk harness](../experiments/logic_first/vnext_utf8_evidence_screen.py)
- [Vim source freeze](../experiments/logic_first/vnext_vim_catalog_freeze.py)
- [Vim external score](../experiments/logic_first/vnext_vim_catalog_one_shot.py)
- [Git unseen-source freeze](../experiments/logic_first/vnext_git_catalog_freeze.py)
- [Git one-shot comparison](../experiments/logic_first/vnext_git_catalog_one_shot.py)

**Code and run outputs are the research artifacts. All fitting and evaluation run on GitHub Actions, not the user's Mac.**
