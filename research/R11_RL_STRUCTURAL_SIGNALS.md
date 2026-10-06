# R11 RL Structural-Signal Diagnostic — 2026-10-06

Authoritative run:
https://github.com/Johnny-Kao/CodecCat/actions/runs/37401037751

## Purpose

R9 isolated five residual RL failures whose truth labels were at raw ranks 10-16. R10 showed generic class balancing does not improve retrieval.

R11 asked whether cheap encoding-specific structural signals contain enough information to justify a narrow RL specialist.

Signals were computed on the first 4096 bytes:
- strict decode validity;
- replacement rate;
- round-trip mismatch.

Candidate encodings:
- UTF-8
- Big5
- EUC-KR
- CP1251
- ISO-8859-1

## Findings

### Big5

Truth Big5 rows:
- n = 12
- strict-valid = **100%**
- mean replacement rate = 0
- mean round-trip mismatch = 0

Other relevant RL rows:
- n = 26
- strict-valid = **50%**

This is useful but not sufficiently specific for a hand-written rule.

### EUC-KR

Truth EUC-KR rows:
- n = 1
- strict-valid = **100%**

Other relevant RL rows:
- n = 37
- strict-valid = **0%**

The signal is very strong directionally, but n=1 is far too small to justify a direct heuristic.

### CP1251 / ISO-8859-1

Both are permissive single-byte codecs:
- CP1251 strict-valid rate is effectively 100% for both truth and non-truth rows.
- ISO-8859-1 strict-valid rate is 100% for both truth and non-truth rows.

Therefore decode validity provides no discriminative value for these classes.

### UTF-8

RL excludes strict UTF-8 by definition, so strict-valid is 0 for both groups.

Replacement severity differs:
- truth UTF-8 rows have much smaller replacement / round-trip damage than many non-UTF8 rows;
- however there are only two RL truth UTF-8 rows.

This may be useful as a model feature, not as a hard rule.

## Decision

Do **not** add direct structural heuristics.

There is enough directional signal to justify one controlled experiment:

**R12 — RL candidate specialist**

The specialist should:
- run only on RL;
- consume existing raw scores plus a small set of structural signals;
- remain linear/small;
- be trained cross-fold;
- be applied only after the existing locked downstream stack;
- require positive paired overall and RL accuracy before any further consideration.

No production behavior changes yet.

CC-MAIN-2026-30 remains untouched.
