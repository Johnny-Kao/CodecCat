# CodecCat — Independent legacy encoding evidence and L3 ablation (2026-11-01)

**Scope:** accuracy-first, reproducible remote GitHub Actions only.  
**Branch:** `research/accuracy-data-gate-v2`. **No changes to main, Alpha, or PyPI.**

## New external real-file source selection

Two independent source-control repositories, **never used for training**:

- `ThomasDickey/lynx-snapshots@7f670781d769903d5a32a4b9eb68175f2e687a26`
- `gnu-mirror-unofficial/gettext@1bc095c8f78b4ff07b48535033e504c126624f3c`

Previous training remains pinned `chardet/test-data@b0c0d206913ba118ffea2f7d306883e7a4a76db7`.

Before running any candidate on these sources, [Actions #41240397820](https://github.com/Johnny-Kao/CodecCat/actions/runs/41240397820) froze **29** source catalogs / **115** raw byte windows, including all eligible non-UTF8 catalogs and six preselected UTF8 controls per source. Frozen source metadata digest: `5b26415a5fbe016dd25789a1a55508d7c5ad84f1912cf753b40002c162ccfe3e`.

The corpus is original source-controlled `msgstr` encoded bytes, with charset declarations and English `msgid` text deliberately excluded. Reference text is obtained by strict decoding with the file's declared charset. Six identical raw windows were excluded. No exact decoded-text overlap with training was observed; near duplicate and source-family overlaps are not exhaustively ruled out. Result: **109** windows, including **61 legacy-encoded** and **48 UTF8** windows. Encodings: `euc_jp` (11), `iso8859-1` (20), `iso8859-15` (5), `iso8859-2` (16), `iso8859-7` (5), `koi8-u` (4), `utf-8` (48).

**Statistical limit:** four windows per catalog are correlated observations. Source count is two, and many codecs have far fewer than 20 examples. These results cannot estimate real-world population prevalence.

## One-shot baseline/external holdout, before inspecting test outcomes

[Actions #41240514666](https://github.com/Johnny-Kao/CodecCat/actions/runs/41240514666) passed with these strict-decoded-text results:

| Model | Correct Unicode / 109 | Exact codec label / 109 | Changes vs stable23 |
|---|---:|---:|---|
| stable23 | **102** | 98 | — |
| full Latin-trained model | 103 | 101 | 3 helps / 2 harms |
| selective Latin routing | 102 | 98 | 2 helps / 2 harms |
| optional UTF8 evidence | **102** | 98 | 0 helps / 0 harms |
| combined selective + UTF8 evidence | 102 | 98 | 2 helps / 2 harms |
| chardet 7.7.0 | **105** | 84 | 3 helps / 0 harms |
| charset-normalizer 3.5.2 | 92 | 58 | 3 helps / 13 harms |

**Cause-level observations** (from saved per-case artifact, original bytes were not extracted into this repository):

- The baseline's seven wrong decoded-text cases: 4 `gettext-tools/po/nn.po` iso8859-1 windows, 2 `gettext-tools/po/ro.po` iso8859-2 windows, 1 `gettext-tools/po/et.po` iso8859-15 window.
- Global Latin specialization corrects two Romanian ISO-8859-2 windows, but makes two previously correct Hungarian `lynx/po/hu.po` windows incorrect by promoting iso8859-16 over iso8859-2. Selective routing does **not** prevent this within-family error.
- `chardet 7.7.0` is better on these original bytes but also misdecodes the four Norwegian Nynorsk ISO-8859-1 windows; chardet's net +3 comes from correcting Romanian (2) and Estonian (1).
- The UTF8 evidence layer preserved all 61 native non-UTF8 decisions and all 48 UTF8 decisions in this set, with **no changed outputs**. This is a regression observation, **not** a proof of no false positives on other encodings.

## Use the existing trainable L3 instead of adding fixture-specific rules

The `LanguageCoherence` L3 code already exists in the installable project, is learned from the original training Unicode text and is deliberately opt-in. We tested it **after** the above external holdout had been inspected, so the following is a **development ablation**, not independent confirmation.

[Actions #41240824828](https://github.com/Johnny-Kao/CodecCat/actions/runs/41240824828):

| Model | Correct decoded text / 109 | L3 calls | L3 changed |
|---|---:|---:|---:|
| stable23 L2 only | 102 | 0 | 0 |
| **stable23 + existing L3** | **103** | 6 | 1 |
| selective Latin + L3 | 103 | 8 | 1 |
| UTF8 evidence + L3 | 103 | 6 | 1 |

For stable23+L3, paired correctness **1 help / 0 harms**, correcting an ISO-8859-15 example. However this data was already inspected and is not a new blind sample. Do not claim that enabling L3 improves global accuracy based on this ablation, and do not change opt-in defaults.

This is a concrete direction for maintainable product differentiation: trainable L2 and language-coherence L3 can be replaced independently, with immutable data manifests, paired correctness evidence, and explicit control of activation. It avoids hardcoding exceptions for Norwegian/Estonian/Romanian/Hungarian.

## Decision

**No model promotion, no release, no PyPI, no main changes.** The extra training-only Latin model has visible cross-language within-family regressions. The UTF8 structural rescue remains opt-in, having caused no changes in this new legacy test suite. Existing L3 is retained opt-in as its single correction is useful exploratory evidence but not decisive.

**Next quality gate:** a truly untouched independent file source with non-UTF8 coverage and reliable source labels, frozen before examining either L3 or UTF8 model behavior. Fix the actual undercovered language/codec families through source-diverse training/holdout separation rather than source-name hardcoded routing or post-hoc threshold tuning. Accuracy first; CPU, RAM, and latency only after correctness can be demonstrated across source origins.

## Exact executable research artifacts

- [Frozen source manifest extractor](../experiments/logic_first/vnext_legacy_po_freeze.py)
- [Pinned original-byte one-shot comparison](../experiments/logic_first/vnext_legacy_po_one_shot.py)
- [Existing L3 trainability ablation](../experiments/logic_first/vnext_legacy_l3_ablation.py)
- [Related previous real-byte evidence](REAL_BYTE_ACCURACY_EVIDENCE_2026-11-01.md)
