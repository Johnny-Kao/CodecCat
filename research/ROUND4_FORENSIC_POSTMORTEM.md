# Round 4 forensic postmortem — no new research run

Date: 2026-10-15. Source: archived GitHub Actions artifact `new-arch-round4-layered-ablation/trace.jsonl` from run [38673401495](https://github.com/Johnny-Kao/CodecCat/actions/runs/38673401495).

## Evidence, independently re-counted from per-case traces (4,096 B)

- 3,117 deduplicated samples.
- Original L1 metric `l1_false_exclusions=153` **omits one unseen training label**. Actual sample-level `survives_l1=false` = **154**.
- Breakdown: UTF-32 **153**, ISO-2022-JP-EXT **1**.
- Of **97** cases where L1 changed a correct L2 prediction into a wrong one, **all 97 are UTF-32 → UTF-16**.
- L3 invoked **3,116/3,117** and changed **zero** final predictions. It has no information not already used in L1.

## Confirmed fault — BOM prefix precedence

The old diagnostic code:
```python
bom=(data.startswith(b"\\xef\\xbb\\xbf"),data.startswith(b"\\xff\\xfe"),data.startswith(b"\\xfe\\xff"))
targets=("utf-8-sig", "utf-16", "utf-16")
```
mistakes UTF-32 LE BOM `FF FE 00 00` for UTF-16 LE `FF FE` because it checks a shorter prefix first. This yields 153 excluded UTF-32 cases, 97 demonstrably harmful prediction transitions. Handling UTF-32 BE `00 00 FE FF` was also missing.

**Safe change:** sort *four-byte* BOMs before two-byte BOMs, and interpret BOM as candidate evidence only when that encoding and decoder policy is supported. Don't force UTF-16 if UTF-32 BOM has matched, even if UTF-32 is not among trained labels. Keep `unknown/unseen label` distinct from `false exclusion from available labels`.

## Next action: contract before benchmark

- Establish a small deterministic pure test suite for all BOM variants (UTF-8, UTF-16 LE/BE, UTF-32 LE/BE), overlap/truncation and invalid-looking prefixes.
- Require L1 to return a *structured result* `(possible, impossible, incomplete, bom_hint, ambiguity)`; it must not secretly pick a single encoding for unsupported/unseen labels.
- Fix metric denominators and separate `truth_excluded`, `truth_unseen`, and `truth_equivalent_decode`.
- L3 cannot repeat full strict-viability checks: only trigger if L2 remains uncertain AND independently new evidence can be computed, or abstain.
- Preserve old experiment outputs and code immutable as forensic evidence. Build corrected implementation separately and test with deterministic unit tests before spending the last GitHub Actions research round.

The error attribution is verified from the trace; a corrected detector is NOT yet validated, and no accuracy improvement is claimed.
