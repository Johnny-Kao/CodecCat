# CodecCat L1/L2 responsibility audit — 2026-11-01

Status: **Completed GitHub Actions SUCCESS**, development diagnostic only.
Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41289347597

## Results

Three source-group folds, 3,086 rows from the same pinned chardet/test-data development source; same residual stable23 parameters, no tuning.

| Metric | Value |
|---|---:|
| Held-out samples | 3,086 |
| Structural-only uniquely possible encoding | 0 |
| Structural possible candidate set >1 | 3,086 |
| Ground truth absent from L1 possible set | 7 |
| Ground truth absent from L2 validated top-three | 221 |
| Baseline strict decoded text correct (audit's comparison function) | 2,830 |
| `structural_first=True` strict decoded text correct | 2,830 |
| L1-first helps / harms / top-one changes | **0 / 0 / 0** |

**Do not claim L1-first improves accuracy.** In this study, prevalidating all candidates yielded exactly the same decisions and correct-text count as the current lazy learned-score order + decoder validation. This is consistent with both paths eventually enforcing the same strict decoder validity for top-three candidates.

Important audit distinction: the earlier training-strategy run reported **2,828** correct for residual, whereas this standalone responsibility audit reports **2,830**, even though both appear to use residual stable23. The runs use somewhat different presentation/decoding comparison semantics or data context. Until a paired-case comparison isolates the discrepancy, **do not claim this is a two-sample accuracy gain**.

## Architecture decision

1. Merely moving structural validation before L2 is **not** an accuracy solution; it can add decoding work without changing classification. Keep current production sequence.
2. Need **distinct new evidence**, not duplicate validation: byte transition legality, syntax-state invariants, reliable BOM/escape/UTF structure, and for intrinsically overlapping codecs, explicitly model ambiguity.
3. The 221 cases with absent truth from validated top-three are useful diagnostics, but include cases where another codec yields the same decoded text. Classify by *wrong text* before prioritizing them.
4. The **seven** cases where the declared truth is missing from L1 possible require data/normalization/partial-prefix debugging before changing L1 (possible aliases, malformed test labels, or incomplete suffix).
5. Do not add per-language heuristics to pass one fixture. Retain retrainable light L2 ranking and a separate model promotion contract.

No change to main, alpha, packaged model or PyPI. No external validation claim. Next development should start from explicit evidence sufficiency tests, not SGD tuning.
