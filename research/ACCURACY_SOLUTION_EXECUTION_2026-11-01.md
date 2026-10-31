# CodecCat — accuracy-first solution execution, 2026-11-01

**Branch:** `research/accuracy-data-gate-v2`  
**Scope:** source isolation, quality-aware training, current competitor evidence. **No CPU/speed/RAM work, no publishing, no main or Alpha changes.**

## Completed, validated steps

1. **Data-contamination repair:** Previous filename-group folds allowed 26 exact decoded documents to cross splits, involving 89 evaluated rows. Replaced experimental cross-validation with source-group + decoded-document connected components before fold allocation (1,069 components; folds 1,029 / 1,029 / 1,028). Cross-split source group, raw SHA and exact decoded-content overlap are asserted absent. **This is not a complete near-duplicate detector.** Prior study: [#41231227499](https://github.com/Johnny-Kao/CodecCat/actions/runs/41231227499).
2. **Strict training-only contrastive samples:** Generate only from labeled *training* bytes that strictly decode and round-trip into alternate codec bytes; never copy evaluated text into training. [#41233258732](https://github.com/Johnny-Kao/CodecCat/actions/runs/41233258732) on content-isolated folds: baseline **2,809/3,086** same-text, full contrast **2,837/3,086**. The full contrast strategy caused CP037 regressions (21→6), so **rejected**.
3. **Remove EBCDIC contrast augmentation rather than add rules:** [#41233450603](https://github.com/Johnny-Kao/CodecCat/actions/runs/41233450603) re-evaluated all 3 arms in one CI run on identical bytes and document-safe folds: baseline **2,811/3,086**, all-pair contrast **2,832/3,086**, **Latin-only contrast 2,838/3,086**, net +27 decoded-text examples. CP037 unchanged at **21/34** in the Latin-only arm; CP500 **13→14/25**. **This is previously studied development data, not final product acceptance.**
4. **Pre-register separate CC0 sources before any model scoring:** Official Tatoeba per-language **CC0 subset** source files; 2026-10-31 exports, eng/fra/rus. No score inspected during frozen source creation [#41233758720](https://github.com/Johnny-Kao/CodecCat/actions/runs/41233758720). Total **560** human-authored sentences, *synthetically re-encoded* one codec per sentence. Frozen selected digest: `6cbf9c040ea05bc295e3f6c4fd9f78bf92371bf6fbf63880d14c27de182f122e`. Archived source SHA-256:
    - eng `c2e2edd063a31cbbc8e416a4858225e934bbb0afff485031821e59f230ccf1ca`
    - fra `cf4509f2f3fd542222be2cf085d0d814ce3f9405263c5746bcc965d7e83dfd08`
    - rus `3b6645bbc18e4e3444ebc9195c4a6c41ff521a8bc4306fec8fcc217ae734db76`
5. **One-shot outside-domain confirmation:** [#41233855912](https://github.com/Johnny-Kao/CodecCat/actions/runs/41233855912) compared two full-corpus fitted stable23 models with `chardet==7.7.0` and `charset-normalizer==3.5.2` on identical CC0-derived bytes. Confirmed 0 exact decoded-text overlap with training. Strict decoded-text correct:
    - baseline **524/560 = 93.57%**
    - Latin-only contrast **526/560 = 93.93%** (**+2 net**)
    - chardet 7.7.0 **554/560 = 98.93%**
    - charset-normalizer 3.5.2 **498/560 = 88.93%**
    - Codec-label accuracy is a separate metric; many valid encodings yield identical decoded text. The candidate is **NOT promoted** because +2 on a synthetic independent source is insufficient to show strong generalizable improvement or match chardet.
6. **Reproducibility:** [#41231691345](https://github.com/Johnny-Kao/CodecCat/actions/runs/41231691345) and separately [#41233192472](https://github.com/Johnny-Kao/CodecCat/actions/runs/41233192472) matched exactly on model input, learned vocabulary/IDF/coefficients/intercept and prediction SHA for folds 0 and 2. Other earlier experiment harnesses produced small differing output counts; until fully explained, quality comparisons are always same-run paired with locked code and source.
7. **Bring-your-own-data trainable framework:** Research-only tool `vnext_custom_retrain_gate.py` takes a file/codec/source-group/split/rights manifest, rejects exact train/validation bytes and decoded-document overlap, exports portable candidate model, compares to incumbent, *cannot* auto-promote. Eight tests passed [#41231473940](https://github.com/Johnny-Kao/CodecCat/actions/runs/41231473940).

## Next directly implemented research

`experiments/logic_first/vnext_cc0_ambiguity_aware_training.py` adds **new genuinely external CC0 training texts**, excluding the 560 previously reserved confirmation sentence IDs. It creates an additional supervised training row only if the encoded data yields a **different decoded text under all other candidate codecs for that language**. This avoids teaching a historical encoding label for bytes that cannot identify that encoding. The comparison uses content-safe held-out chardet/test-data folds. **Previously viewed dev folds and the earlier CC0 560 confirmation cannot be re-labeled as new blind evaluation.**

Execution: [GitHub Actions #41234071867](https://github.com/Johnny-Kao/CodecCat/actions/runs/41234071867). Final disposition must follow its actual paired development results, not merely CI success.

## Release decision and product requirements

Current verdict: **NO PROMOTION**. Maintain Alpha `v0.1.0a1` and frozen default. Do not publish to PyPI or tune CPU/RAM/speed.

**Product differentiator:** a repeatable user-controlled training and updating protocol, with immutable data/version manifests, source-disjoint validation, independently scored candidate-vs-incumbent comparisons and explicit rollback. Making `fit()` callable alone does not finish this product.

New candidate acceptance requires independently sourced, **real naturally encoded file bytes** with reliable ground-truth provenance, larger representative coverage, accepted per-family regressions, and current competitor comparison. Synthetic CC0 reencoding supplies useful independent language text but is **not** representative of naturally deployed file encoding frequencies.
