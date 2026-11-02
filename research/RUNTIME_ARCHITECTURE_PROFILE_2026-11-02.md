# CodecCat runtime architecture — cost attribution, 2026-11-02

CI: https://github.com/Johnny-Kao/CodecCat/actions/runs/41370068467 (PASS).
Pinned source: chardet/test-data b0c0d206913ba118ffea2f7d306883e7a4a76db7; held-out grouped fold 0, first 500 fixtures. Model trained outside timed loop.

| Stage | Median process CPU per document |
|---|---:|
| scikit-learn TF-IDF feature transform | 2.69582 ms |
| sparse linear scoring plus sorting | 0.15952 ms |
| complete rank (structural gates + vectorize + score + validated candidates) | 3.00030 ms |
| all-class structural evidence (not normal hot path) | 0.28503 ms |

Process high-water RSS: 270.5 MiB, including Python, sklearn and training. Not a clean inference-only RAM measure.

**Decision:** The feature transform is the critical runtime target, not the matrix scorer. The measurements are sequential samples, therefore ratios are approximate and cannot be treated as a precise flamegraph. Keep training in Python; develop a narrow native feature-extraction prototype with exact TF-IDF and ranked-candidate parity before altering stable API or published wheels.

**Status:** The prototype submission was interrupted by tooling controls and did not land in this branch. No native speed gain has been measured. Main and release unchanged.
