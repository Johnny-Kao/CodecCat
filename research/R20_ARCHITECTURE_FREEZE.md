# R20 Architecture Freeze

Date: 2026-10-07
Branch: `staging/initial-package`

## Decision

The CodecCat development architecture is frozen after R19C and R20B.

Frozen execution:

```text
route once
-> HMT768 once
-> baseline 518d once
-> baseline scorer/downstream
-> 50% training-quantile raw-margin gate
-> conditional U/RH full-B 886d using cached HMT
-> RL-only R12 specialist
-> final rank
```

Frozen suitability parameters:

- scorer C = 0.5
- R12 threshold = 0.65
- gate fraction = 0.50
- S3 = 0.02
- HMT = 768
- full-B routes = U/RH

## Evidence

R19C:
- C=0.4 -> 378/418
- C=0.5 -> 378/418
- C=0.6 -> 378/418
- accuracy range = 0 hits

R20A:
- route mismatch = 0
- baseline feature mismatch = 0
- full-B feature mismatch = 0

R20B:
- 378/418 = 90.4306%
- escalated = 249/418
- prediction mismatch = 0
- escalation mismatch = 0
- cached path median ~1189.9 us/sample vs reference ~1569.7 us/sample in the same run

Runs:
- R19C: https://github.com/Johnny-Kao/CodecCat/actions/runs/37538628158
- R20A: https://github.com/Johnny-Kao/CodecCat/actions/runs/37549807634
- R20B: https://github.com/Johnny-Kao/CodecCat/actions/runs/37552050491

## Holdout rule

No further development-set tuning is allowed before the fresh-crawl gate.

A fresh holdout may be evaluated only after:
1. clean runtime implements this frozen architecture;
2. one release model is trained on all eligible development data;
3. that artifact is serialized and fingerprinted.

If fresh-holdout evidence is used to change the architecture or any frozen suitability parameter, the holdout becomes development data.

`CC-MAIN-2026-30` remains excluded from tuning and is not to be inspected for candidate selection.
