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
