# L3 repeated reliability matrix — 2026-11-01

Status **GitHub Actions SUCCESS**: https://github.com/Johnny-Kao/CodecCat/actions/runs/41311453160

## Experimental design

Ubuntu 22.04 and Ubuntu 24.04 GitHub Actions, pinned training/test repositories, Python and sklearn. Compare full vs residual stable23 L2 training. Refit each strategy three times per operating system and evaluate same 21 nonidentical normalized-text uchardet fixture files. No tuning or production modifications.

**Important**: These are 12 deterministic repeated training-and-evaluation combinations on the **same 21 documents**, not twelve independent corpora or randomized seeds.

## Reproduced results

| Strategy | L2 correct | L2 + L3 correct | Helps | Harms | Repeats |
|---|---:|---:|---:|---:|---:|
| Full | 19/21 | 20/21 | 1 | 0 | 6/6 identical decisions |
| Residual | 20/21 | 19/21 | 0 | 1 | 6/6 identical decisions |

The residual-model L3 harm was **Big5 -> CP273** in all six evaluations. Both OS environments gave identical choices within each strategy. L3 marginal timing median around 4.2–4.5 microseconds on all 21 samples (mostly no-op; not conditional L3 invocation cost). Timing should not be used as an inference benchmark without isolated repeated timed calls and process controls.

## Interpretation

The L2 training-selection strategy changes candidate order and hence what L3 verifies. Existing L3 can improve one full-training configuration and harm a different residual configuration. It cannot be promoted unconditionally. There is no evidence of nondeterministic behavior or platform instability in this matrix, and the 21 fixtures do not establish independent-source accuracy.

## Decision

1. Stop repeated deterministic runs of the same fixtures as a means of seeking accuracy improvement.
2. Keep production L3 opt-in. Do not patch one Big5 exception or tune a magic margin.
3. Next product-grade work is a validation contract: model+L3 provenance fingerprints, pinned data, independent confirmation and explicit no-regression/rollback gates.
4. A future structural detector may produce new universally valid evidence but must preserve legitimate code pages and prove impact on independent samples.

Code: `experiments/logic_first/l3_repeated_reliability.py`. Artifacts from both OS jobs include raw per-case records. No changes to main, Alpha or PyPI.
