# CodecCat

CodecCat is an independent character-encoding detector under active development.

The project is designed around three priorities:

- **speed** — bounded byte work, one feature/scoring pass, and small fixed-shape downstream decisions;
- **simplicity** — a compact runtime architecture instead of a large heuristic stack;
- **maintainability** — runtime code, fitted model state, research harnesses, and benchmarks are kept separate.

CodecCat is not a fork of another detector.

Its two primary external comparators are:

1. `charset-normalizer`
2. `chardet 7`

## Current research result

Canonical cross-validated held-out result:

- 365 / 418 correct Top-1 predictions
- 87.3206% Top-1
- charset-normalizer reference on the same canonical development benchmark: 87.1429%

This is **cross-validation evidence**, not yet a release-model accuracy claim.

The current Python/NumPy performance research converged at P15. P16 found no further material exact-equivalent low-risk micro-optimization.

## Package status

The clean package runtime is being reconstructed on:

`staging/initial-package`

Current structure:

```text
src/codeccat/
  api.py
  features.py
  model.py
  runtime.py
```

The runtime accepts an immutable fitted `RuntimeBundle`. A single release model is deliberately not bundled yet: the existing canonical result uses four fold-specific held-out models, and none of those should be misrepresented as a deployable final model.

Before the first release CodecCat will:

1. prove the clean runtime is exactly equivalent to the locked P15 research path;
2. benchmark CodecCat / charset-normalizer / chardet 7 on identical held-out bytes;
3. train one final release model on all eligible development data;
4. evaluate that frozen artifact on a new independent holdout;
5. only then ship the model with the package.

## Documentation

- [Project plan](research/PROJECT_PLAN.md)
- [Production runtime specification](docs/PRODUCTION_RUNTIME_SPEC.md)
- [Benchmark protocol](docs/BENCHMARK_PROTOCOL.md)
- [Release model protocol](docs/RELEASE_MODEL_PROTOCOL.md)
- [Research plan](research/RESEARCH_PLAN.md)
- [Current handoff](research/HANDOFF.md)
