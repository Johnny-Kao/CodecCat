# CodecCat model freeze and external-source validation — 2026-11-06

## What is completed

**Release-model reproducibility (build output): PASS.**

Previous CI workflows retrained `stable23` while assembling Wheels, and repeated runs produced different model hashes despite a pinned source revision. Instead of requiring stochastic retraining to be bitwise deterministic, release build jobs now load the **exact reference material from the published `v0.1.0a4` release** and verify both the archive and model SHA256 before packaging.

- Frozen reference archive: `https://github.com/Johnny-Kao/CodecCat/releases/download/v0.1.0a4/native-reference-v0.1.0a4.tar.gz`
- Archive SHA256: `e3dd654817fb0e3324910b1a39af9bc19185400582d5a066c533d7cc5dd81c8c`
- Bundled model SHA256: `16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b`
- [Six-platform CI PASS #42153391865](https://github.com/Johnny-Kao/CodecCat/actions/runs/42153391865)
- [Root-wheel CI PASS #42153902552](https://github.com/Johnny-Kao/CodecCat/actions/runs/42153902552)

This solves **release artifact model consistency**, not deterministic retraining from raw text. The model is still unchanged from published `v0.1.0a4`.

## Source-disjoint evaluations performed

Both datasets were selected and pinned to an immutable public repository commit before scoring the published Wheel, alongside chardet `7.7.0` and charset-normalizer `3.5.2`.

| Source | Commit | Samples | CodecCat correct text | chardet | charset-normalizer |
|---|---|---:|---:|---:|---:|
| GNOME/gtk PO originals | `7c578d79eb8347b14a1717bfcc0084674a43c562` | 328 | 296 | 328 | 326 |
| BYVoid/uchardet test originals | `4e685757780cb3c652fc6c9ec759f62888969ec9` | 102 | 94 | 96 | 90 |

- [GTK 5-fold CI PASS #42153394720](https://github.com/Johnny-Kao/CodecCat/actions/runs/42153394720), [script](../experiments/gtk_blind_holdout.py)
- [Multi-encoding 5-fold CI PASS #42153686436](https://github.com/Johnny-Kao/CodecCat/actions/runs/42153686436), [script](../experiments/uchardet_blind_holdout.py)
- Each fold groups windows by source document to prevent same-document cross-fold leakage; the result files preserve input SHA256, original encoding and per-case outcome.
- GTK provided only UTF-8 labeled files after exclusion and strict label decoding, so it cannot establish cross-encoding performance.
- The uchardet source provided **35 standard-Python-supported encodings across 57 source files**. The `viscii` and `euc-tw` samples were explicitly skipped because no Python standard codecs are available.
- No classifier modification or rule optimization was performed based on these evaluations.
- This is **source-disjoint**, but a full training-text near-duplicate/semantic overlap audit is not yet established. Related text-document genres and multiple windows per source document reduce independence.
- One run per source on GitHub-hosted hardware; CPU/latency measurements are diagnostic only.

## Later overlap audit — correction to independence interpretation

A subsequent pinned-source exact-overlap audit [CI #42235598046](https://github.com/Johnny-Kao/CodecCat/actions/runs/42235598046) replicated all 328 GTK and 102 uchardet benchmark windows. **33/102 uchardet windows are exact full-byte matches** to examples from the historical L2 training corpus; **50/102** flag sampled-span near-overlap. These sets intersect in 26 windows: **57/102 have at least one overlap flag**. The previous assumption that all 33 exact matches were among the 50 sampled-span candidates was incorrect. GTK had no detected raw-byte or sampled-span matches under this screen. See [full audit caveats](EXTERNAL_HOLDOUT_OVERLAP_SCREEN_2026-11-06.md).

**Correction:** although the uchardet repository itself was distinct, the evaluated data were not fully independent of training. The 94/102 score below is descriptive, not unbiased held-out accuracy. Do not characterize uchardet as a fully independent blind test. The GTK result also remains a limited, previously observed single-genre diagnostic, not proof of semantic independence.

## Clear outcome

**These were Alpha v0.1.0a4 results, not current Stable v0.1.0 results.** The Alpha missed 32/328 GTK texts compared with zero chardet misses and 8/102 on the uchardet sample compared with 6 chardet misses. Later pinned-source overlap analysis invalidated the characterization of the uchardet cases as independent. Calling CI PASS a product-accuracy PASS would be misleading. The released Stable 0.1.0 has separately measured 324/328 GTK and 95/102 uchardet correct texts; these remain previously inspected development diagnostics, not independent confirmation.

**Engineering priorities derived from the untouched results:** inspect the 32 GTK UTF-8 failures and the per-encoding uchardet misses (notably Latin-15, cp1252, Shift-JIS), then make changes using *a separate new confirmation source*. Previously inspected heldouts become development fixtures after the results have been examined; they must not be presented as pristine blind tests for future revisions.

**The earlier Alpha evaluation did not publish a new model.** Subsequently `codeccat==0.1.0` became available through PyPI; distinguish that released Runtime from the earlier pinned Alpha wheel when citing accuracy. The evaluation and overlap research did not publish any new models.
