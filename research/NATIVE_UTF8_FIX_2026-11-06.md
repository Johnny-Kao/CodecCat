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

## Reproducible staged benchmark

- [Final six-platform wheel parity CI #42148598734](https://github.com/Johnny-Kao/CodecCat/actions/runs/42148598734): all 6 platforms PASS, reliability guards PASS.
- [Staging same-byte five-corpus + external GLib CI #42148882888](https://github.com/Johnny-Kao/CodecCat/actions/runs/42148882888): complete PASS.
- Pinned test code: [five-corpus benchmark](../experiments/five_corpus_benchmark.py), [new external GLib pair gate](../experiments/native_utf8_external_gate.py).

Five existing corpora, 843 cases (Python 3.12, Ubuntu 24.04):

| Metric | Candidate Native | chardet 7.7.0 | charset-normalizer 3.5.2 |
|---|---:|---:|---:|
| Correctly decoded Unicode text | 819 (97.15%) | **829 (98.34%)** | 698 (82.80%) |
| Exact historical encoding name | **781 (92.65%)** | 712 (84.46%) | 583 (69.16%) |
| Total CPU on timed 843 calls | **453 ms** | 482 ms | 667 ms |
| Median latency range across five corpora | 0.57–0.61 ms | **0.25–0.43 ms** | 0.46–0.62 ms |
| p95 latency range | **0.84–0.89 ms** | 1.39–1.79 ms | 1.90–2.87 ms |
| Process peak RSS range | 66–69 MiB | 61–64 MiB | **40–44 MiB** |

**Paired before/after decoded-text counts on the same fixed source datasets:**

| Source pool | Published v0.1.0a3 | Candidate |
|---|---:|---:|
| chardet/test-data, model-training overlap (633) | 622 | **623** |
| charset-normalizer upstream fixtures (15) | 14 | **14** |
| Vim/Git PO windows previously inspected (195) | 169 | **182** |
| **Total (843)** | **805** | **819** |

**New frozen source confirmation:** `GNOME/glib@d34a927488b8b664cbd305dc880e86f34f8ca467`. After freezing the commit before scoring, sampled 285 distinct original translation-file byte windows with PO charset declaration masked. Same-file, same-case baseline v0.1.0a3 vs candidate results: **227 → 271 correct Unicode texts (+44)**, 44 helps, **0 harms**. This is a single new source domain; it is not population-level accuracy or proof of no regression across every legacy codec.

Important measurement constraints:
- The five-corpus cases overlap source texts used for model training and previous investigation; percent values cannot be marketed as blind population accuracy.
- GLib documents were not part of earlier research in this repository, but exact near-duplicate/text-family overlap with all model-training data has not been fully excluded; treat this as source-level confirmation, not independently audited contamination-free holdout.
- Timings come from a **single GitHub-hosted runner run**; cross-run variation matters. Do not claim tiny percentage CPU improvements are statistically significant.
- The "v0.1.0a4 candidate" benchmark uses a staging wheel with the same decoder fix and model contents. Official tagged v0.1.0a4 wheels are separately built, verified, and published after final checks; do not conflate it with a measurement of those final assets.
- The remaining 13/195 external incorrect cases include Latin-family and Big5 confusions not solved by UTF-8 evidence. No hard-coded document-specific exceptions were introduced.

## Decision

After six-platform CI and staged same-byte/regression/source-level testing, the UTF-8 improvement is suitable for an experimental GitHub Alpha. **No PyPI publishing authorized.** Larger mixed naturally encoded blind corpora and source-contamination analysis remain necessary for general accuracy claims.
