# CodecCat

CodecCat is an independent character-encoding detection package under active development.

Its goal is to provide a detector that is:

- **fast** — minimize repeated byte scans, feature extraction, and scoring work;
- **simple** — keep the runtime architecture small and understandable;
- **maintainable** — separate the stable package core from training and research harnesses;
- **competitive** — benchmark directly against `charset-normalizer` and `chardet 7` on the same held-out data.

CodecCat is not a fork or patch of either comparator.

## Current research milestone

The canonical held-out evaluation currently reaches:

- **365 / 418** correct Top-1 predictions
- **87.3206% Top-1**
- charset-normalizer reference on the same canonical benchmark: **87.1429%**

The Python/NumPy runtime optimization program converged at **P15**. P16 found no further material low-risk micro-optimization winner.

## Current direction

The next phase is package productionization:

1. consolidate the accepted research architecture;
2. implement a clean installable `src/codeccat/` package;
3. reproduce the canonical result from package code;
4. run a formal three-way benchmark:
   - CodecCat
   - charset-normalizer
   - chardet 7
5. prepare the first standalone release.

See [research/PROJECT_PLAN.md](research/PROJECT_PLAN.md) for the full project plan.

Active research branch:
`research/charset-normalizer-e2e-baseline`
