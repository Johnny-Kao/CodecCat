# Runtime stabilization decision (2026-11-02)

## Architecture choice
**Freeze the Python reference pathway before native acceleration.** Training and experimentation remain Python. Native C is paused until the execution graph, fidelity gates and memory baseline have been fixed.

## Stable operational contract
1. **Training boundary:** source-group labeled bytes -> feature profile and numeric learned weights. No training during normal `detect` calls.
2. **Model boundary:** portable, numeric-only archive; schema version, class labels, TF-IDF vocabulary, IDF, weights, intercept, fingerprint. Preserve legacy Stable23 compatibility. Fail closed for unrecognized profiles.
3. **Inference boundary:** bytes -> prefix capped at 4096 -> decisive structural gates -> normalized TF-IDF vector -> learned score order -> strict per-candidate decode -> up to three valid ranked candidates. No inferred probability from uncalibrated score.
4. **Public boundary:** `CodecCat.detect/decode`, BOM policy and opt-in language model remain stable; shipped Alpha defaults to Stable23.
5. **Reproducibility boundary:** same archive produces identical encoding, candidates and ambiguity fields across repeated in-process calls and fresh Python processes.
6. **Performance boundary:** measure cold startup, warm steady-state CPU, peak RSS for inference-only fresh process, and feature/scoring/gate breakdown separately. Benchmark exact same held-out cases and inputs; reject regressions before merging.
7. **Accuracy boundary:** per-source disjoint evaluation, source overlap audit and per-encoding paired regressions. Model tuning never uses final external holdout.

## Evidence
- Runtime profile https://github.com/Johnny-Kao/CodecCat/actions/runs/41370068467: TF-IDF transformation 2.696 ms median, sparse scoring/sorting 0.160 ms, full rank 3.000 ms (500 samples, stage timings sequential).
- Steady-state gate https://github.com/Johnny-Kao/CodecCat/actions/runs/41370960064: 300 held-out documents, three runs and fresh processes with hash seeds 0,1,17; all identical. Three consecutive 300-document CPU totals 0.8064/0.8548/0.8196 s.
- Existing portable archive round-trip evidence https://github.com/Johnny-Kao/CodecCat/actions/runs/41368432314.

## Execution decisions
- **Current Alpha remains Stable23.** Binary12 is an experimental, serializable candidate, not a default.
- **Do not begin C port yet.** First establish isolated inference-only memory and fully compatible Python feature extractor; then profile independently. If Python optimization already removes most cost, no C port may be needed.
- No changes to published v0.1.0a2, main or PyPI.
