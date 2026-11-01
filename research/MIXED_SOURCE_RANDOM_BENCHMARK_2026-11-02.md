# Mixed-source seeded benchmark — 2026-11-02

GitHub Actions: https://github.com/Johnny-Kao/CodecCat/actions/runs/41320018205 — SUCCESS.

## Sampling

- Pinned chardet/test-data development corpus, chardet-v7.5.0 and charset-normalizer-v3.5.2.
- Seed 20261102; source/encoding/fold strata; sampled about 35% per chardet stratum, minimum 3.
- 1,283 unique raw-byte samples: 1,275 from chardet corpus, **only 8** from charset-normalizer's 18 labeled sample files. Ten candidate charset-normalizer files were excluded as byte-identical to records in chardet dataset. This means the source balance is **99.4% chardet**; calling this a balanced mixed-source independent test would be inaccurate.
- CodecCat fitted to two other grouped folds for chardet samples, or full chardet corpus for charset-normalizer samples; no random bytes are fabricated.
- Chardet 7.5 and charset-normalizer 3.5.2 are installed competitor versions; external source provenance and semantic/near duplicates are not independently audited.

## Overall (paired identical documents)

| Engine | Correct | Accuracy | CPU inference p50 | p95 | Process RSS peak |
|---|---:|---:|---:|---:|---:|
| CodecCat residual stable23 | 1150/1283 | 89.63% | 2.9783 ms | 5.1577 ms | 444.46 MiB |
| chardet 7.5.0 | 1241/1283 | 96.73% | 0.5148 ms | 4.2552 ms | 227.18 MiB |
| charset-normalizer 3.5.2 | 1057/1283 | 82.39% | 0.8247 ms | 3.6059 ms | 226.99 MiB |

Source breakdown:
- chardet 1275: CodecCat 1143, chardet 1233, charset-normalizer 1049.
- charset-normalizer 8: CodecCat 7, chardet 8, charset-normalizer 8.

**Critical interpretation**: Large chardet sample stratum favors chardet-v7.5; the 8 records from the other source cannot counterbalance it. A random order or stratified sample inside biased sources does not establish superior generalization. Results disagree with chardet 7.0.1 prior 3086-case comparison; version changed, and sampling changed, so do not attribute the full delta to version alone.

Process RSS includes imports and all CodecCat fold training (4 fitted models). It is not isolated inference-only memory. CPU medians are process CPU for individual calls, not wall clock throughput. Exact source-duplicate exclusion is in the selection itself; semantic overlap remains unaddressed.

Full per-case predictions, sampled SHA256 manifest and metrics are available in the Actions artifact `mixed-random-sample-profiles`.

## Decision

Do not claim CodecCat beats chardet or charset-normalizer overall. Release-readiness requires an independent, provenance-audited and source-balanced evaluation set with no training overlap. Optimize latency/RAM only after establishing accurate fair inference-only profiles.
