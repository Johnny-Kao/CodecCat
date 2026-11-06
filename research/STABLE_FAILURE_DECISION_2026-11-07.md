# Stable 0.1.0 encoding-quality decision — 2026-11-07

## Decision

**Do not lower the default UTF-8 rescue threshold from 3 to 1 or 2.**
Do not change the default numeric model, packaging, public API, or PyPI.
This is supported by the paired tests below, not by an assumption that
all Latin encodings are uniquely identifiable.

## Previously inspected development sources

- GTK: Stable 324/328; 4 UTF-8 windows guessed as ISO-8859-1.
- uchardet: Stable 95/102; 4 ISO-8859-15, 2 cp1252, 1 Shift-JIS
  false-text outputs, with an overlap-contaminated historical corpus.
- The 11 diagnosed cases decode exactly as expected *under the chosen
  codec*; the problem is choosing the wrong codec, not a second-step
  Unicode decoder corruption.
- The existing rule requires >=3 non-ASCII Unicode chars, strict UTF-8
  validity, length >=64, no BOM, and no disallowed controls.
- The 4 GTK failures contain 2, 2, 2 and 1 such Unicode chars, respectively.
- Existing paired GTK/uchardet: threshold 1 yields GTK 328/328 and
  uchardet 95/102, with no observed harms **on those inspected sources**.

Evidence:
[CI 42241797564](https://github.com/Johnny-Kao/CodecCat/actions/runs/42241797564)
and [CI 42242829202](https://github.com/Johnny-Kao/CodecCat/actions/runs/42242829202).

## Why the apparently successful change is not safe

60 explicitly synthetic Windows-1252 samples were constructed so their
**intended CP1252 characters** happen to encode to valid UTF-8 bytes.
These are *adversarial counterexamples*, not population accuracy estimates.

| Rescue threshold | Correct of 60 | Correct baseline cases harmed |
|---|---:|---:|
| Published baseline / threshold 3 | 24 | 0 |
| Threshold 2 | 11 | 13 |
| Threshold 1 | 0 | 24 |

This proves that the proposed general correction has harmful cases.
It does not establish their prevalence in real production files.

Evidence: [CI 42244038675](https://github.com/Johnny-Kao/CodecCat/actions/runs/42244038675).

## Newly preregistered CPython confirmation

Protocol was checked in **before any scoring**:
[CPython fixture protocol](./CPYTHON_ENCODING_CONFIRMATION_PROTOCOL_2026-11-07.md).

- Pinned CPython revision
  `f64e51563a85667446793ad61ea521d005ceb171`, 15 CJK
  encoding/reference pairs, 30 complete files.
- Against pinned historical 3,086 source files, **zero** full/prefix
  exact overlaps or sampled-block near-overlap candidates triggered.
  This is a finite raw-byte screen, not proof of semantic independence.
- Stable 26/30, chardet 29/30, threshold 1/2/3 each 26/30. On this
  particular source, counterfactual UTF-8 rescue changes no result.
- Four Stable mismatches:
  - big5hkscs, euc_jisx0213, shift_jisx0213: **not present among model
    trained classes**, so these cannot be repaired by a different
    ranking among existing class labels.
  - cp949: class supported, but predicted cp1006. This remains a true
    supported-class misranking worth later focused investigation.
- CPython sample sizes are tiny (one file per legacy encoding) and
  **do not cover cp1252 / ISO-8859-15**. Thus it is not sufficient to
  approve lowering the UTF-8 threshold or to claim universal quality.

Evidence:
[CI 42243748064](https://github.com/Johnny-Kao/CodecCat/actions/runs/42243748064)
and [CI 42244274798](https://github.com/Johnny-Kao/CodecCat/actions/runs/42244274798).

## Engineering next action and stop condition

1. Keep threshold 3 and published Stable 0.1.0 unchanged.
2. Treat missing model classes as coverage decisions, not a bug in the
   strict decoder. Do not expand the model without training rights and
   independent multi-source confirmation.
3. The remaining supported-class cp949 example and legacy ISO/cp1252
   ambiguities can inform future model research, but not new one-off
   overrides tuned to seen failures.
4. Stop filing repeated diagnostic PRs. Revisit model changes only when
   a new rights-reviewed, independent dataset spans UTF-8, CJK and
   Western single-byte encodings, with paired benefits/harms and
   runtime speed/memory evidence.
5. Quality analysis is technically documented; **model release remains
   blocked by dataset rights review and insufficient independent
   Western-encoding coverage**.

All research CI files are metadata-only and use ephemeral external
checkouts. They do not upload source documents or replacement models.
