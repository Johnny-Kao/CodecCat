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

## Additional never-before-used Gawk / Bash source: one-shot confirmation

Frozen original Gawk/Bash PO bytes **before any model evaluation** in [#41241122649](https://github.com/Johnny-Kao/CodecCat/actions/runs/41241122649):

- Sources: `gnu-mirror-unofficial/gawk@455c99c010eafca5729081954d23059e3cbca571`; `gnu-mirror-unofficial/bash@9439ce094c9aa7557a9d53ac7b412a23aa66e36b`.
- Source digest `7e11e0d76121103b6e30ba23f3f64e0416f15b65214efc76fc7b50edf098300e`.
- 79 original-byte windows: 31 non-UTF8 (19 ISO-8859-1, 4 ISO-8859-15, 4 ISO-8859-2, 4 ISO-8859-8) + 48 UTF8; no exact source-training text duplicates found.

Frozen-candidate one-shot scores in [#41241223413](https://github.com/Johnny-Kao/CodecCat/actions/runs/41241223413):

| Model | Correct decoded text / 79 | L3 activations | Prediction changes |
|---|---:|---:|---:|
| stable23 L2 | 72 | 0 | 0 |
| stable23 + unchanged L3 | **72** | 5 | **0** |
| selective Latin + unchanged L3 | 72 | 5 | 0 |
| UTF8 evidence + unchanged L3 | 72 | 5 | 0 |
| chardet 7.7.0 | **75** | n/a | n/a |
| charset-normalizer 3.5.2 | 67 | n/a | n/a |

**Outcome:** The previous development-corpus L3 +1 improvement did **not** replicate on untouched Gawk/Bash; L3 remains OFF by default. No automatic publication, no general accuracy superiority claim.

## Ground-truth discovery: a strongly suspicious stale charset declaration

Before blaming all model errors, verify the correctness of the source's claimed encoding. Some single-byte codecs (especially ISO-8859-1) can decode *any* bytes strictly, so `decode(..., "strict")` succeeding does not establish the historical encoding label.

A source-only (no model predictions) audit [#41241463665](https://github.com/Johnny-Kao/CodecCat/actions/runs/41241463665) detected a **strongly suspicious metadata mismatch** in `gnu-mirror-unofficial/gawk/po/it.po`:

- Source declares `ISO-8859-1` but **all four** selected translation-byte windows decode strictly as UTF8;
- All four UTF8 decodings contain non-ASCII text;
- Decoding the exact bytes using the declared ISO-8859-1 label yields **38** common `Ã[continuation-byte]`/similar mojibake-pair patterns across the sampled windows.
- The other seven legacy-declared Gawk/Bash catalogs in the frozen selection did not show this signal.

This is **high confidence evidence of a possible stale/mismatched encoding declaration**, not definitive proof and not permission to relabel. In the original frozen holdout, all four Gawk Italian windows count as incorrect Unicode for stable23 *and* chardet because the declared ISO-8859-1 text is the reference; both predict UTF8. The original **72/79** and **75/79** scores are preserved unchanged. A separate future sensitivity analysis may distinguish audited vs ambiguous labels, but must **never silently drop inconvenient benchmark cases or claim newly blind results**.

### Implemented product-level remediation in custom retraining

- `vnext_custom_retrain_gate.py` now screens custom manifest entries where a declared single-byte encoding strictly decodes, but original bytes also have strong UTF8 evidence and produce repeated UTF8-as-Latin mojibake sequences under the claimed label.
- **Default: fail closed before fitting.** Users may explicitly mark an individual source entry `"charset_provenance_reviewed": true` *only if they independently verified the source*. Such accepted warnings are recorded with source path, byte hash, declared codec and evidence counts in `source_ledger.suspected_charset_mismatch_reviewed` and are explicitly tagged **self-attested**, not independently validated.
- Genuine Latin-1 and ordinary ASCII controls remain valid; user-supplied provenance string alone does not disable the check.
- [GitHub Actions #41241798209](https://github.com/Johnny-Kao/CodecCat/actions/runs/41241798209) **12/12 tests passed**, including file path, licensing metadata, train-validation overlap, portable model and incumbent comparisons.
- [Pinned original training-data audit #41241894839](https://github.com/Johnny-Kao/CodecCat/actions/runs/41241894839): **0 of 3,086** existing corpus examples triggered this *conservative specific* detection. This is not a guarantee of all source labels being correct; the filter intentionally has limited recall.

**Release decision remains NO-GO:** preserve `main`, existing Alpha and PyPI. The new label-integrity gate improves custom trainability quality and maintainability, not demonstrated automatic inference accuracy. Only GitHub Actions was used for testing.

## Added exact code

- [Gawk/Bash preregistration](../experiments/logic_first/vnext_new_gnu_source_freeze.py)
- [Gawk/Bash untouched-source one-shot L3 comparison](../experiments/logic_first/vnext_gnu_unseen_l3_confirm.py)
- [Source-only encoding declaration audit](../experiments/logic_first/vnext_native_po_charset_audit.py)
- [Custom-retrain label checks](../experiments/logic_first/vnext_custom_retrain_gate.py)
- [Training manifest regression tests](../tests/test_vnext_custom_retrain_gate.py)
- [Original training corpus quality audit](../experiments/logic_first/vnext_training_label_integrity.py)
