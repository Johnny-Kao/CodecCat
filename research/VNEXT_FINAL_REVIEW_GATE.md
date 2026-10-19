# CodecCat vNext — Final Review Gate (Draft)

Status: **NOT READY FOR FINAL SIGN-OFF**. All changes remain on `staging/initial-package`. Do not change `main`, publish releases, or update PyPI.

## Accepted engineering evidence

- Independent new-architecture L1 structural contracts and endianness regression tests pass.
- Trainable ordered 2–3-gram TF-IDF / SGD L2 is implemented; training accepts `(bytes, encoding, source_group)`; model bundles are for trusted sources only.
- Legacy control-character L3 verifier was a net loss and is disabled. Do not present an active third-stage verifier as validated.
- Developer-corpus grouped validation reaches 84.12% exact label on a 3,003 sample paired benchmark. Another 3,086-sample grouped ablation yields 84.61% under capped easy-example removal; these have different effective sample populations and **must not be compared as a single longitudinal trend**.
- Independent-repository uchardet data: 113 labeled samples, 77 remain after exact byte and decoded-text overlap removal. On those 77, CodecCat **48 exact / 56 same decoded text**, charset-normalizer **46 / 55**, chardet **34 / 34**. Tiny 1-case text advantage is not broad superiority.
- [GitHub Actions deterministic replay #38692069569](https://github.com/Johnny-Kao/CodecCat/actions/runs/38692069569): two fresh full fits on ordered immutable data produce exactly equal vocabularies, learned coefficients, intercept and external predictions. Fingerprint `936eb0331a0d0ac447b0f6c6135bec7cd397962544d053cda3220358dacba85c`. External benchmark result 48/77 exact, 56/77 same text.

## Must close before asking the owner for final review

1. **Independent coverage:** add one more externally sourced, responsibly licensed, nonduplicated test corpus with enough samples per codec. Use isolated train/test provenance; lock the test prior to design decisions. No tuning based on uchardet diagnostics.
2. **False certainty:** validate short ambiguous ASCII/Latin labels, invalid streams, partial prefix boundaries, Unicode aliases and unsupported encodings. Expose abstention policy and risk/coverage curve. Never infer inherently unidentifiable historical labels.
3. **True language L3:** implement a separately trained decoded-language coherence candidate reranker, gated to genuinely uncertain cases; test help, harm, invocation rate, and net cost on source-disjoint data. If it harms quality, leave off.
4. **Actual L2 accuracy:** further improve Baltic/Estonian / code-page confusions by acquiring training content from separate sources, not ad hoc holdout-specific rules.
5. **Runtime acceptance:** isolate inference-only p50/p95, peak RAM, CPU, memory cost of loaded model and training; include representative and worst-case file sizes. Do not use Mac M5 for performance claims.
6. **Stable reproducibility:** pin dependency versions and dataset commits, store the training record manifest + schema hash, dataset license/source ledger, and compare predictions across repeated full GitHub Actions runs. Earlier independent runs yielded 49 vs 48 exact labels; repeated same-run training is deterministic, but the historical one-case discrepancy is still not explained.
7. **Safety and packaging:** avoid eager legacy module imports into vNext; move to a self-contained package or verify old imports are inaccessible. Enforce version compatibility, no untrusted joblib deserialization.

## Reviewer handoff condition

Only propose owner sign-off when every row above has concrete test artifacts and no failed gate. This file is a **blocker ledger**, not a substitute for executing remaining work or an assertion of completion.

## Execution update — 2026-10-15

**New completed engineering gates:**
- Legacy import isolation: changed the public `codeccat` namespace to lazily resolve existing legacy API symbols, so `import codeccat.vnext_strong_l2` no longer imports `codeccat.model`, `codeccat.api`, or `codeccat.model_io`. Existing legacy symbols remain importable. CI [#38693465177](https://github.com/Johnny-Kao/CodecCat/actions/runs/38693465177): 20 passed.
- A **non-pickle numeric-only `.ccnext` archive** is implemented with explicit model schema, TF-IDF vocabulary, IDF, coefficients and intercept. The decoder validates file entries, size limits, shape, finiteness and schema; it must only be used on trusted files until broader malicious-archive fuzzing. Fixed a reconstruction error where sublinear TF was omitted. Both binary and multi-class score consistency tests now pass. CI [#38693993021](https://github.com/Johnny-Kao/CodecCat/actions/runs/38693993021): 21 passed. Earlier failing CI checks caught real defects and were fixed, not suppressed.
- A **third dataset-origin diagnostic** has been added: official charset-normalizer `test_full_detection.py` fixtures, pinned `Ousret/charset_normalizer` SHA `4b335594aee803e04b62e8ffa1a5a2b7ebd4d33c`. [CI #38694177443](https://github.com/Johnny-Kao/CodecCat/actions/runs/38694177443) passed. Of 18 official source fixtures, 10 exact-byte overlaps and 9 exact-text overlaps were found with training; **only 8 samples remained nonoverlapping**. On those 8, same-text decode: **CodecCat 7/8, chardet 7/8, charset-normalizer 8/8**. Exact codec label: **CodecCat 6/8, chardet 7/8, charset-normalizer 8/8**. Since these are the competitor's own tests, do **not** call this independent neutral superiority comparison. On all 18, same-text: CodecCat 16, chardet 15, charset-normalizer 18.
- This third source rules out declaring model quality universally settled. It also highlights test-data provenance overlap. Prior uchardet 77-nonoverlap holdout is small and differences are within 1–2 examples.

**Still blocks final owner review:** larger and more representative independently curated, licensed and isolated multilingual corpus; calibration of confidence and abstention; trainable decoded-language coherence verification (L3 remains off); inference-only RAM/CPU and cross-architecture results; historical discrepancy of one exact label between earlier runs requires per-example comparison. This is engineering evidence, **not signoff**. No changes to `main`, frozen R21, PyPI or releases.

## Corrected priority: prove basic learning on current corpus first (2026-10-15)

The user correctly pointed out that independent-source score discussions were premature without separating in-sample training fit from held-out generalization. Added dedicated `experiments/logic_first/vnext_fit_audit.py` and GitHub Actions `vnext-fit-audit.yml`. The audit measures true label identity and **incremental decoded-text equivalence** (critical for truncated 4,096-byte prefixes), separately and without conflating train fit with validation.

- Original fitted-all model: 3,028/3,086 exact-label; initial decoded-text score had a **diagnostic bug** from treating truncated multibyte prefixes as final. Correcting this metric exposed **43 actual decoded-text errors**, not 116.
- Found actual runtime bug: BOM fastpath used `data.decode(..., 'strict')` on truncated 4,096-byte prefixes, causing false abstention for BOM-prefixed UTF-8. Changed the production L1 fastpath to use a strict incremental decoder with `final=False` for truncated streams. Added regression test.
- [Corrected in-sample audit #38701679827](https://github.com/Johnny-Kao/CodecCat/actions/runs/38701679827) passed, model fit on all 3,086 examples: **3,039/3,086 = 98.477% exact label**, **3,053/3,086 = 98.931% identical decoded text**, **33 actual wrong-text cases**; 14 historical label-only mismatches have identical text. Capped easy-example training fit 3,034 exact and 3,049 same-text, with 37 different-text errors. L2 full-sample fit is therefore stronger than capped-residual for in-sample accuracy; capped-residual retains only a modest source-held-out benefit and should not be treated as the universally superior default.
- Remaining actual in-sample confusion includes ISO-8859-16→ISO-8859-2 (5), CP1252→ISO-8859-15 (5), CP037↔CP500 (8). Many ambiguous one-byte codecs require decoded-language evidence rather than more BOM rules.
- This **does not mean 98.9% held-out/generalization**. Source-group-held-out cross-validation was ~84%, and must improve on *existing development folds* before broadening external acceptance. Freeze these revised split definitions and no leakage.
- All changes on staging; core tests [#38701661584](https://github.com/Johnny-Kao/CodecCat/actions/runs/38701661584) passed. No main/release changes.

Next **mandatory priority**: explain + close source-group validation gap on the same current dataset, with error per-label/source-group and clean train/validation partition, before any new external corpus work.

## Newly selected untrained corpus — 2026-10-15

Created a frozen external benchmark from **amrali-eg/UnicodeTestSuite** SHA `80684bd405505b069cfec4e07493c8713a26f41a`, originating from independently sourced UDHR multilingual passages and their generated encodings; the generator creates versioned documented Unicode/legacy text samples. This is not a natural file-frequency distribution: multiple samples share UDHR source documents, and Unicode variants are overrepresented. Selected up to 300 filepaths using a stable hash BEFORE scoring, and excluded exact byte/decoded-text overlap with the chardet/test-data training samples. Tested CodecCat capped-residual, **chardet 7.6.0** and charset-normalizer 3.4.4 using the same 4,096-byte prefix, exact-label and incremental decoded-content equivalence rules.

[CI Run #38705403056](https://github.com/Johnny-Kao/CodecCat/actions/runs/38705403056) SUCCESS, n=300:

| Model | Exact encoding label | Same decoded text |
|---|---:|---:|
| CodecCat vNext | 121/300 = 40.33% | 158/300 = 52.67% |
| **chardet 7.6.0** | **132/300 = 44.00%** | **175/300 = 58.33%** |
| charset-normalizer 3.4.4 | 103/300 = 34.33% | 144/300 = 48.00% |

**Result: NOT PASS.** On this externally sourced synthetically re-encoded corpus, the current CodecCat model is **5.67 percentage points behind chardet 7** in decoded-content correctness. Historic comparisons against chardet 5.2.0 are obsolete for a latest-market claim. The n=300 set overrepresents UTF-16/UTF-32, UTF-8, GB18030 and BOM variants; inspect per-label trace to separate Unicode false-negative pathways, labeling semantics and genuine byte-language errors. This run did not modify model parameters or train on the new holdout.

**Decision:** do not claim overall benchmark victory or ask owner for final audit. Focus existing-corpus source-group generalization and diagnostic correctness, then run locked multiple external corpora, including current chardet 7.

## Randomized multi-distribution external test (2026-10-15)

[GitHub Actions #38705931581](https://github.com/Johnny-Kao/CodecCat/actions/runs/38705931581) successfully ran `experiments/logic_first/vnext_multidist.py` with locked seed `20261015`. Training only on previously defined chardet/test-data with capped L1 easy-case removal, and zero external examples introduced into training. External candidate sources were uchardet and generated UnicodeTestSuite; exclude exact matching training bytes/text; deduplicate bytes across the pooled candidate set. Remaining **1,068 distinct candidates**: 77 uchardet, 991 UnicodeTestSuite. Output artifact contains `report.json`, `traces.jsonl`, and the fixed selected IDs `selection.json`.

Two distinct sampling distributions:

| Fixed sampling strategy | n | CodecCat exact / text | chardet 7.6 exact / text | charset-normalizer exact / text |
|---|---:|---:|---:|---:|
| **Source × encoding-family stratified**, max 50 per cell | 227 | 108 / **127 (55.95%)** | 119 / **147 (64.76%)** | 93 / 119 (52.42%) |
| **Pooled random**, max 350 | 350 | 122 / **147 (42.00%)** | 131 / **161 (46.00%)** | 115 / 148 (42.29%) |

Sample provenance and distribution matter strongly: stratified 227 = uchardet 77 and generated suite 150, families 54 East-Asian / 87 single-byte / 79 Unicode / 7 other. Pooled 350 = generated suite 325 and uchardet 25, of which 278 Unicode. The pooled fixture frequency is **not** production/natural frequency and generated suites are correlated by original UDHR document. The two samples overlap, so do not treat them as statistically independent benchmark replications. Byte equality and decoded-text equality remove only exact duplicates, not near duplicates.

**Fail gate:** CodecCat remains behind chardet 7 in both randomized distributions; not ready for owner's final review. Engineering direction: repair structural Unicode false exclusions and group-specific ambiguity, then test against the now-locked sample IDs without selecting hyperparameters on the evaluation set; a separate unseen confirmation set is required after model changes. Do not claim success based only on prior chardet 5.2 or in-sample 98.93% results.

## L1 escape designation implementation and fixed-sample ablation (2026-10-15)

Added `src/codeccat/vnext_escape.py` with conservative, validated ISO-2022 JP/KR initial escape-designation recognition. Integrates before statistical scoring, only accepts labels already supported by the trained model, preserves incomplete/unknown/noisy escape cases by falling back to L2. Added unit regressions for both codecs, untrained codec and terminal escape; core contract [Actions #38724512604](https://github.com/Johnny-Kao/CodecCat/actions/runs/38724512604) SUCCESS.

Compared the exact same frozen random external selections (seed 20261015) to the prior no-escape result; [Actions #38724622552](https://github.com/Johnny-Kao/CodecCat/actions/runs/38724622552) SUCCESS:
- 227 source-family-balanced: CodecCat decoded-text **127/227 before and after**, exact label **108→107** (one lost).
- 350 pooled random: CodecCat decoded-text **147→149**, exact label **122→123**.
- chardet 7 stays at 147/227 and 161/350 decoded-text. **No stable, broad external win established**. Variations potentially from ISO-2022 evidence; inspect traces and structural false positives before promoting to a default.
- Explicitly note: these existing frozen external datasets are validation diagnostics and should not be used for repeated optimization selection. New unseen confirmation must decide whether structural path stays enabled.

Pending final release blockers remain: Unicode variants, actual script consistency L3, independent coverage, benchmark gap to chardet 7, and cross-architecture resource testing. No main or release changes.

## Latest implementation checkpoint — 2026-10-15 (L1 structural breadth)

Added deterministic, conservative trained-label-gated paths:
- `src/codeccat/vnext_escape.py`: initial ISO-2022-JP/KR escape designations with strict incremental decoding; unsupported, unrelated terminal escape or incomplete designation falls through instead of forcing a codec.
- `src/codeccat/vnext_unicode_structure.py`: no-BOM UTF-32 LE/BE detection based on strongly asymmetric four-lane NUL patterns on ASCII-heavy inputs, with strict incremental decode viability and trained-class eligibility. Evaluated before UTF-16 byte-parity heuristic to avoid nested-encoding false positives.
- New unit tests include positive JP/KR, UTF32 LE/BE, untrained class, false terminal escape, UTF16 non-interference. [CI core run #38724953721](https://github.com/Johnny-Kao/CodecCat/actions/runs/38724953721) SUCCESS.

**Fixed 20261015-seeded external multi-distribution diagnostic (no training on evaluation sources):**
- Before any new L1 rules: source×family-balanced 127/227 same-text; pooled 147/350.
- ISO2022 escape only: balanced 127/227; pooled 149/350.
- ISO2022 + UTF32 no-BOM: [CI #38725072714](https://github.com/Johnny-Kao/CodecCat/actions/runs/38725072714) balanced **127/227**, pooled **148/350**. Comparators unchanged chardet 7 at 147/227 and 161/350. **There is no broad verified gain** from these easy structural additions.
- These 2 random distributions overlap and are development diagnostics once revisited; they cannot certify generalization. The correct next step is paired source/family breakdown and prioritization, not continued hand-tuning from these exact rows.

Final status: **NOT READY FOR RELEASE OR OWNER SIGNOFF**. Remaining crucial gaps: general Unicode/script coverage, source-heldout accuracy, trained language-based selective reranker, multilingual representative benchmark with additional held-out sources, cross-architecture and resource gate. No changes to main/R21/releases.

## Latest source-stratified candidate-recall root-cause study (2026-10-15)

Do not continue adding arbitrary L1 rules or promise a working L3 without top-k coverage. Updated `experiments/logic_first/vnext_multidist.py` to emit per-source/family paired error matrices, top-3 candidate coverage, exact mistake pairs and separate score metrics. [Validated action #38738657459](https://github.com/Johnny-Kao/CodecCat/actions/runs/38738657459) succeeded.

Frozen external source × family sample, n=227, decoded text:
- East Asian 54: CodecCat 15, chardet 7 17; **correct encoding in CodecCat top 3 only 15/54**. GB18030 frequently misranked as KOI8-T or CP1006. This is an L2 class/feature coverage issue, not an L3 rescoring issue.
- Single-byte 87: CodecCat 58, chardet 7 76; **correct top-3 candidate 61/87**. Largest paired competitor loss (18 text hits) concentrated here.
- Unicode 79: CodecCat 47, chardet 7 47; correct top-3 47/79. Repeated BOM/generic/explicit endian label-semantic differences warrant *separate* comparison; do not silently normalize away BOM or count invalid decoding as success.
- Other 7: both 7.
The pooled-random distribution (350) is heavily overrepresented by synthetic Unicode suite and must not be interpreted as a real-world production prevalence.

**Engineering gate change:** increase top-k *recall* and source-group performance before implementing the L3 reranker. New controlled independent-of-external-holdout experiment `experiments/logic_first/vnext_l2_feature_ablation.py` with workflow `vnext-l2-feature-ablation.yml`; evaluates 2–3 ngrams original, 2–4 ngrams, 1–3 ngrams and class-balanced SGD using 3-fold group-disjoint training, with exact, decoded-text and top-three coverage. [Action #38738942106](https://github.com/Johnny-Kao/CodecCat/actions/runs/38738942106) launched; results must be checked before any trained-runtime change. It may be slow due to 12 group-trained full sparse-text classifiers and is **not evidence of improved accuracy until PASS and measured**.

Never tune on external datasets used here, preserve frozen rows/seed, do not modify main or R21. Final review gate remains not passed.

### Completed four-way L2 grouped feature ablation

[GitHub Actions #38738942106](https://github.com/Johnny-Kao/CodecCat/actions/runs/38738942106) **SUCCESS** on all 3,086 corpus rows, same frozen source-group three-fold split, group disjoint training/test and no external tuning:

| Feature specification | Exact labels | Identical decoded text | Truth in validated top3 |
|---|---:|---:|---:|
| baseline 2-3gram min_df2 | 2610 | 2809 | 2879 |
| 2-4gram min_df1 | 2585 | 2784 | 2853 |
| **1-3gram min_df1** | **2626** | **2833** | **2886** |
| balanced class weights 2-3gram | 2518 | 2754 | 2782 |

The 1–3 gram alternative gained **+16 exact labels, +24 same-decoded text, +7 truth-in-top3** on this source-group development set; the gain is modest (24/3086=0.78 pp). Despite increased top3 recall, the much larger external source coverage gap still exists. Neither 2–4 gram nor balanced class weights should be promoted. **Do not yet replace the packaged 2–3 gram runtime**: the frozen evaluation sources have been repeatedly inspected and cannot independently confirm this choice. Run a separately locked unseen-source confirmation, test per-source degradation and archive compatibility first. Current portable schema explicitly expects 2–3 ngrams; a model change without migration would break load_portable.

## Paired 1–3gram validation by encoding family (2026-10-15)

Built `experiments/logic_first/vnext_paired_features.py` and dedicated free Ubuntu workflow. Unlike simple aggregate comparisons, both candidates use exactly matching immutable source-group folds and every evaluated byte sample is paired, recording gains, losses, top3 and family/codec breakdown.

**[GitHub Actions #38747332529](https://github.com/Johnny-Kao/CodecCat/actions/runs/38747332529) SUCCESS**, source-held-out development n=3086:
- Original 2–3gram: 2,800 correctly decoded text; candidate 1–3gram: **2,825**; 40 new wins, 15 new losses (net +25).
- Every group-disjoint fold improves: fold0 1041→1053, fold1 738→742, fold2 1021→1030.
- Single-byte family n=877: 715→737 (+22); East Asian n=203: 190→192 (+2); Unicode n=1544: 1523→1523 (no net change); Other n=462: 372→373 (+1).
- Not a uniformly dominating model: CP437 declines 27→25, CP850 38→37, UTF16-BE 153→152, UTF7 141→140. Per-label regression veto / risk tolerances must be specified independently of this observed result.
- This paired run reports 2800→2825, whereas the earlier broad 4-way experiment reported 2809→2833. **These nominally identical configuration discrepancies require investigation** (e.g. corpus fixture state, candidate source version and scoring implementation) before any signoff. Do not merge the two report numbers or claim fully stable reproducibility.

**Decision:** 1–3gram is a credible staged candidate in the current dev corpus, not yet promoted; previous portable archive schema is 2–3gram only, so live model replacement would invalidate the archive contract. Another genuinely withheld corpus is still required before final acceptance, plus targeted regression checks for adversely affected encodings. No changes to main or published artifacts.

## 2026-10-15: paired score reproducibility and provenance

Found two previously quoted group-CV aggregates for apparently matching feature configs: broad feature ablation 2809→2833 and later paired study 2800→2825. Added an executable check to `experiments/logic_first/vnext_paired_features.py` that runs the **original** `vnext_l2_feature_ablation.evaluate` on the **same models and same fold data** as the paired scorer, and asserts equal old/new decoded-content counts for every fold. Both [verification #38750588808](https://github.com/Johnny-Kao/CodecCat/actions/runs/38750588808) and [provenance replay #38751011213](https://github.com/Johnny-Kao/CodecCat/actions/runs/38751011213) passed: fold 0 1041→1053; fold 1 738→742; fold 2 1021→1030; total 2800→2825, +40 wins / −15 losses. The difference from older 2809→2833 is **not yet causally explained**, so do not erase it or claim fully reconciled historical reproducibility.

Paired report now saves SHA256 of ordered dataset metadata, model coefficients/intercepts/vocabulary for each fold/config and exact NumPy/scikit-learn versions. Current dataset fingerprint `1c947f5a767b35127a7d9428db2ccac69a5991b1e948e9e544fe19eee1a05d02`, numpy 2.5.3, sklearn 1.9.1. Artifacts saved to workflow runs for inspection. Keep 1–3 gram as experimental; no default runtime/model portable schema change until independent verification and per-class regressions are handled. No main or release changes.

## Opt-in 1–3gram production adapter readiness (2026-10-15)

Implemented non-default `TrainableDetector.fit(..., feature_profile="experimental13")` for source-group dev-confirmed 1–3 byte-character ngram TF-IDF (min_df=1, 90k features), preserving original `stable23` as default. Added separate numeric portable archive schema `vnext-l2-tfidf-ordered13-sgd-huber-v2`; `load_portable` accepts both explicitly versioned 2–3 and 1–3 manifests while keeping strict param matching. No code-executing pickle required for portable loading. New tests verify opt-in fit, portable save/load inference parity, and profile rejection; existing archive tests unchanged.

[Contract CI #38752831710](https://github.com/Johnny-Kao/CodecCat/actions/runs/38752831710) **SUCCESS**. This is engineering readiness **only**: independent benchmark acceptance, full dataset training size/memory/performance, no-regression thresholds and historical dev-score disagreement still block default promotion, release, or owner signoff. Do not represent this as a product accuracy gain.

## External paired 1–3gram opt-in confirmation (2026-10-16)

Tested the actual `TrainableDetector.fit(..., feature_profile="experimental13")` against unchanged default `stable23` on identical frozen external rows. No cross-source fixture is added to training; preserve original corpus versions and fixed `seed=20261015`. [CI #38774702099](https://github.com/Johnny-Kao/CodecCat/actions/runs/38774702099) **SUCCESS**:
- Source × family stratified 227: stable23 **127/227 same-text**, experimental13 **136/227**, chardet 7.6 **147/227**. Paired model replacement wins 14 / loses 5; exact labels 108→116.
- Pooled fixture-random 350: stable23 **148/350**, experimental13 **155/350**, chardet 7.6 **161/350**. Paired wins 13 / loses 6; exact labels 123→129.
- This is **directionally encouraging**, and narrows the observed gap to chardet 7 from 20 to 11 correct text cases (stratified) and 13 to 6 (pooled). Both selections overlap and are repeatedly examined; **not new untouched confirmation**. Do not claim statistically independent replications or unbiased natural prevalence.
- Opt-in portable numeric format and roundtrip tests previously passed. Keep `stable23` default until truly new third-source blind confirmation, resource/cross-arch regression gates and regression thresholds are satisfied. Do not update main, release or PyPI.

## 2026-10-19: annual candidate retraining protocol implemented and exercised

Added `experiments/logic_first/vnext_annual_retrain.py`, `.github/workflows/vnext-annual-retrain.yml`, and `research/VNEXT_ANNUAL_RETRAINING_SOP.md`. SOP defines the repeatable sequence: provenance/license and group-based data partition → stable L1 → ordered unigram/bigram/trigram-feature candidate training → source-disjoint paired dev scoring → untouched-source validation → per-family regression → CPU/RAM/architecture checks → **explicit owner review**. The actual implementation uses *combined* 1–3-gram TF-IDF for `experimental13`; independently weighted per-ngram-family extraction has **not** been implemented and must not be claimed as such.

**[2026 annual run #39181320719](https://github.com/Johnny-Kao/CodecCat/actions/runs/39181320719) SUCCESS**: pinned corpus 3,086 accepted examples, 1,124 source groups, 88 labels. Both candidate models fit, exported numeric portable bundles, and passed representative save/load inference parity. Stable23 has 45,000 vocabulary features, 13,685,936 bytes; experimental13 90,000 features, 24,580,570 bytes. SHA256:
- stable23 `1ba665dc3eddc0dff7e38e7526121d8e80192d9778061fc956ae07d3bcc9ae94`
- experimental13 `be875ef2842f12429f6a5784952787ed3c8687bd0dc2974277cb67e30`
- ordered 2026 input metadata SHA256 `6e76b3f45510cadd5b3e83aa9915bddbc6d3474fb5cbc5f2522a0b9a5c330d04`.

GitHub Actions branch-specific workflow_dispatch may not be addressable via `gh workflow run` until GitHub registers the workflow on the repository default branch; **do not modify main just to make it dispatchable**. A staging push trigger permits testing the workflow. Each future year requires updating **new approved data source snapshots and blind holdouts**; the year field by itself only changes labels and does not fetch new data. No auto release, no silent default change, no main/R21 changes. Current annual-run artifact retention = 30 days, so archive externally for year-over-year history.
