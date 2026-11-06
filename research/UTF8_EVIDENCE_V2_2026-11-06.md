# UTF-8 Evidence v2 — Released v0.1.0a5

## Root cause

The previous Native decoder required at least **16 non-ASCII Unicode codepoints** before its strict-UTF-8 evidence could overturn a legacy single-byte model ranking. This excluded genuine UTF-8 text containing only a few accented or non-Latin characters, causing incorrect Unicode output despite fully valid UTF-8 bytes.

The change lowers this threshold to **3**, retaining strict whole-buffer UTF-8 validation, minimum 64 input bytes, BOM precedence, rejection of C0/C1 control characters, classifier UTF-8 support, and opt-in L3 separation. It never replaces invalid UTF-8 silently.

## Staging paired results (model frozen, not release marketing accuracy)

All tests use the same shipped v0.1.0a4 model bytes and original source bytes, with one change in the Python decoder:

| Source | Published a4 | v2 staging candidate | Gains | Observed harms |
|---|---:|---:|---:|---:|
| GTK original translation windows | 296/328 | 324/328 | 28 | 0 |
| GLib translation windows | 271/285 | 283/285 | 12 | 0 |
| 35-encoding uchardet test files | 94/102 | 95/102 | 1 | 0 |
| Previously used five mixed corpora | 819/843 | 826/843 | 7 | 0 |
| **New pinned GNOME Software translation windows** | **198/225** | **216/225** | **18** | **0** |

[Four-pool staging paired CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42156811989) and [new-source paired comparison](https://github.com/Johnny-Kao/CodecCat/actions/runs/42157052821) passed. The new source is `GNOME/gnome-software@d693edda74a6ae3a5360a6d19ab558c2fd3ac8cb`; it was selected and pinned before scoring and has 225 valid non-ASCII translation windows. The paired CI explicitly verified **identical default model SHA256** `16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b`.

[Six-platform v2 staging wheel CI](https://github.com/Johnny-Kao/CodecCat/actions/runs/42156575255) passed on Linux, macOS and Windows, x86-64 and ARM64.

## Limitations

- These are **source-specific developer test sets** and not representative population accuracy estimates.
- More aggressive UTF-8 evidence may misinterpret deliberately ambiguous legacy-encoded bytes. The synthetic CP1252/ISO-8859-1 negative controls show that some byte streams are fundamentally undecidable, and the heuristic is not proof of the original encoding.
- The new GNOME Software source tests a translation-file domain; genuine messy real-world files and source independence still need ongoing evaluation.
- No retraining or re-tuning of the shipped model occurred.
- This result is an **experimental Alpha improvement**, not a Stable signoff.

## Confirmed published release

[v0.1.0a5 six-platform Tag CI #42157726285](https://github.com/Johnny-Kao/CodecCat/actions/runs/42157726285) passed on Linux, macOS and Windows, both x86-64 and ARM64. All Wheels contain the exact same frozen model SHA256 `16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b`.

[Post-release published-Wheel benchmark #42158308518](https://github.com/Johnny-Kao/CodecCat/actions/runs/42158308518) passed: a5 decoded **826/843** text samples correctly (**97.98%**), versus chardet **829/843 (98.34%)** and charset-normalizer **698/843 (82.80%)**. CodecCat exact encoding labels: **787/843 (93.36%)**.

These numbers are specific to this development benchmark; they do **not** certify general real-world accuracy or Stable readiness. The five-corpus source mix includes 633 training-source samples. The newly pinned GNOME Software paired evaluation remains the stronger source-specific confirmation of improvement, but not representative population proof.

