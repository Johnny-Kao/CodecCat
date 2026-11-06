# Native UTF-8 evidence correction — 2026-11-06

**Scope:** Conservative Unicode correctness improvement in the native decoding layer. Not a new model or retraining. This report accompanies CodecCat Native `v0.1.0a4`.

## Root cause

In five mixed fixed corpora, 26 / 195 external Vim/Git translation windows were decoded incorrectly by v0.1.0a3. Per-case hashed traces identified:
- 22 UTF-8 inputs misidentified as Latin, CP1250, CP273, HP-Roman8 or abstained.
- 3 ISO-8859-1 inputs mistaken for CP1250.
- 1 Big5 input mistaken for CP1250.

The existing numeric model ranks plausible but wrong single-byte codecs above UTF-8 in some sufficiently long real files. Since single-byte codecs can strictly decode almost arbitrary bytes, validation alone cannot reject them.

## Correction

The native public decode path now allows a **conservative UTF-8 rescue** only if:
- Input is complete and at least 64 bytes.
- There is **no recognized BOM**, so UTF-8/UTF-16/UTF-32 signature handling remains authoritative.
- Full strict UTF-8 decoding succeeds.
- At least 16 genuinely non-ASCII Unicode codepoints appear.
- No unexpected C0/C1 control codepoints appear.
- UTF-8 is a supported model class, and the optional L3 mode is not in use.

Rescued outcomes are marked `ambiguous=True` with reason `utf8_evidence`; no fabricated confidence or model retraining. Invalid bytes continue to fail strict decoding. A prior naive test wrongly overwrote six already-correct UTF-8 BOM examples; the final guard explicitly excludes BOM inputs.

## Final tagged release verification

- [Final tagged six-platform CI #42150439965](https://github.com/Johnny-Kao/CodecCat/actions/runs/42150439965): 6/6 PASS; the distributed wheels are tied to immutable `v0.1.0a4`.
- [Python 3.10–3.14 + optional legacy CI #42149882615](https://github.com/Johnny-Kao/CodecCat/actions/runs/42149882615): PASS.
- [Root-wheel CI #42149894376](https://github.com/Johnny-Kao/CodecCat/actions/runs/42149894376) and [public API CI #42149906701](https://github.com/Johnny-Kao/CodecCat/actions/runs/42149906701): PASS.
- [Post-release three-detector five-corpus test #42151051728](https://github.com/Johnny-Kao/CodecCat/actions/runs/42151051728): PASS, downloaded published v0.1.0a4 Wheel.
- [Paired published a3 vs published a4 on GLib #42151536276](https://github.com/Johnny-Kao/CodecCat/actions/runs/42151536276): PASS, both packages carry the **same default model SHA256**.

### Five fixed corpora: published v0.1.0a4, 843 cases

Python 3.12 / Ubuntu 24.04; chardet 7.7.0; charset-normalizer 3.5.2. Every detector sees identical bytes.

| Metric | Released CodecCat v0.1.0a4 | chardet 7.7.0 | charset-normalizer 3.5.2 |
|---|---:|---:|---:|
| Correctly decoded text | 819/843 (97.15%) | **829/843 (98.34%)** | 698/843 (82.80%) |
| Exact normalized encoding name | **780/843 (92.53%)** | 712/843 (84.46%) | 583/843 (69.16%) |
| Total CPU, 843 timed calls | **601 ms** | 801 ms | 1,061 ms |
| p50 latency range across five corpora | 0.75–0.79 ms | **0.43–0.74 ms** | 0.72–1.00 ms |
| p95 latency range | **1.11–1.18 ms** | 2.21–2.81 ms | 3.04–4.67 ms |
| Whole-process peak RSS | 66–69 MiB | 61–64 MiB | **40–44 MiB** |

These latency/CPU numbers are single-run observations on hosted CI, not statistical guarantees. The overall 843-case evaluation includes CodecCat model training data.

### Paired before/after decoded-text recovery

Both published wheels embed `default.ccm` with **SHA256 `16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b`**, so the model is held constant.

| Source pool | Published a3 | Published a4 |
|---|---:|---:|
| chardet/test-data, overlaps training (633) | 622 | 622 |
| charset-normalizer pinned labeled fixtures (15) | 14 | 14 |
| Vim/Git PO windows previously inspected (195) | 169 | **183** |
| **Total (843)** | **805** | **819** |

The new previously unexamined source is `GNOME/glib@d34a927488b8b664cbd305dc880e86f34f8ca467`, frozen before scoring. For 285 distinct original PO windows with charset declarations masked, **227 correct in a3 → 271 correct in a4**: 44 improvements, **0 regressions**. This demonstrates source-specific gain, not general population accuracy or representative legacy-encoding coverage.

### Model reproducibility caveat found during release QC

An earlier staging six-platform run generated a bundled model with SHA256 `95fb5f1e61970d294051b1e34f09162643e0d74bfbf5d54ea71b07a32d1fefd2`. The final tagged release embeds `16073306...`, matching the published a3 model exactly. The corpus/source revisions were pinned, but rebuilding the model across CI runs is **not reliably byte-reproducible**. Thus the provisional staging measurements (781 labels, 453 ms total CPU and 182/195 Vim/Git) must **not** be presented as measurements of the released model or as a clean paired baseline.

The published wheel assets are immutable and have verified checksums, but future release pipelines should reuse a content-addressed frozen model snapshot rather than silently retrain a new model in every build.

### Limits and unresolved errors

- The five-corpus development benchmark includes training and previously inspected samples; it does not establish a statistically representative general accuracy.
- GLib source-specific evidence uses a new upstream repo, but near-duplicate or document-family overlap with training has not been exhaustively audited.
- The released a4 still decodes **12/195** Vim/Git windows incorrectly; further fixes require case-level root-cause classification and a separate fresh verification source.
- No document-name-specific rules, unapproved model promotion, or PyPI publication.


## Decision

After six-platform CI and staged same-byte/regression/source-level testing, the UTF-8 improvement is suitable for an experimental GitHub Alpha. **No PyPI publishing authorized.** Larger mixed naturally encoded blind corpora and source-contamination analysis remain necessary for general accuracy claims.
