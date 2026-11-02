# Top3 candidate selection — decision record, 2026-11-02

## Executed
- Batch six weights, unchanged Binary TF alpha 0.0003 model: https://github.com/Johnny-Kao/CodecCat/actions/runs/41458297532 (PASS).
- Discriminability audit with the same frozen grouped folds: https://github.com/Johnny-Kao/CodecCat/actions/runs/41458580180.

## Findings
- Original: 2897/3086 correct (93.88%).
- Control/non-printable-quality Top3 rerank weights 0,0.1,0.5,1,2,4: best observed 2899/3086 (93.94%), 2 fixed, 0 broken, 3 selections altered. This is selected on the development corpus; do not promote without independent evaluation.
- 189 wrong originally; 149 recoverable with a *perfect* Top3 reranker, 40 absent Top3.
- Among recoverable 149: 123 have equal control penalty; 26 have lower penalty in correct candidate; 143 have equal decoded-text length; only 6 have differing decoded lengths.
- Correct candidate score disadvantage median 0.194; p90 0.734 (raw uncalibrated model scores).
- The generic control-character penalty cannot distinguish 123/149 recoverable mistakes. It also typically cannot overcome the existing model-score margin.
- Do not assume 149 can be fixed through language coherence alone. Candidate encodings may yield plausibly printable text and have no obvious structural difference. Independent language/script evidence, markup declarations, or trained cross-fitted secondary ranker are plausible separate tests, not yet proven.

## Decision
Keep the released v0.1.0a2 default model and the experimental alpha=0.0003 base unchanged. No new reranker deployed. Stop searching scalar penalties on this development set. Future architecture work should use distinct evidence and source-separated validation, including semantic binary abstain layer. Do not treat development metric improvement as real-world superiority.
