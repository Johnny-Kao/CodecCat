# Top-3 error study — CodecCat 2026-11-02

Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41356216597 (PASS).

- Valid grouped-fold examples: 3,086; mistaken decoded texts in this run: 256.
- Correct text among available ranked top three: 155 errors (60.5%).
- Of these, correct text at rank 2: 121; at rank 3: 34.
- Correct text absent from top three: 101 (39.5%).
- Score margin between winner and runner-up among error cases: median 0.2665; P90 1.1536 (raw modified-huber model scores; not calibrated probabilities).
- Previous run #41355988589 found 258 errors, 153 recoverable and 105 absent; do not aggregate statistics across runs. Further cross-run reproducibility control is needed.

Interpretation: many errors are candidate *ordering* decisions, but an oracle reordering with ground-truth labels would leak the correct answer. No accuracy gain has yet been validated for a deployable reranker. For about 40% of current errors, even perfect Top-3 reordering cannot help; these demand stronger features/coverage or appropriate ambiguity reporting.

Next proposed controlled experiment: try a **single generalizable feature or L2 ranking adjustment** with source-group disjoint validation and paired per-encoding loss reports. Reject if external-source regressions or latency regressions outweigh gains. No unvalidated per-codepage heuristics.

Research only; main/Alpha/PyPI unchanged.
