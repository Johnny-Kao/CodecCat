# External batch accuracy tournament — 2026-11-01

Status: **GitHub Actions SUCCESS; development diagnostic, NOT independent confirmation**.

Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41278594040

Original sources pinned: `BYVoid/uchardet@4e685757780cb3c652fc6c9ec759f62888969ec9` and `chardet/test-data@b0c0d206913ba118ffea2f7d306883e7a4a76db7`.

## Results

57 usable charset-labeled uchardet fixtures; **33 byte-identical** to the pinned development training corpus; evaluate only 24 nonidentical fixtures for this diagnostic.

| Detector | Strict same Unicode text | Exact encoding label |
|---|---:|---:|
| CodecCat stable23 residual, fitted at run time | **23/24 (95.83%)** | 20/24 |
| chardet (installed from PyPI during run; version not pinned) | 22/24 (91.67%) | 19/24 |
| charset-normalizer (installed from PyPI during run; version not pinned) | **24/24 (100%)** | 20/24 |

CodecCat had 1 wrong decoded-text result in Japanese (4/5 correct), while charset-normalizer recovered 5/5. chardet was 4/5 in Japanese, and 1/2 in Chinese; per-file case detail is available in the Actions artifact `external-batch-tournament-report/cases.jsonl`.

3 test-area files skipped (C harness and Python-unsupported VISCII / EUC-TW). No independent CP037/CP500 documents were identified in this fixture set.

## Interpretation

- **Cannot claim competitive win or loss statistically:** just 24 samples from one competitor fixture repository, with strong lineage reuse concerns. Exact-byte filtering catches only identical blobs, not content transcodes.
- The previously used Hungarian fixtures ISO-8859-2 and CP1250 appear in the 33 exactly overlapping samples: those previous two successes **were not independent external evidence**.
- Filename-declared labels have not been independently authenticated.
- A detector may correctly decode the text while returning an alternative encoding label; metrics intentionally separate the two.
- The comparison uses a newly fitted stable23 residual detector, not the serialized alpha artifact. Dependencies for chardet and charset-normalizer were not pinned in this particular run. Treat results as exploratory, not reproducible version-locked acceptance.
- No performance measurements or promotion attempted.

## Decision

Keep `main`, Alpha, and PyPI unchanged. The next batch should target independently originated original-source documents (not common character-detector fixture sets), provenance verification, and document-level deduplication of transcodes; CP037/CP500 remains a critical gap. Do not repeatedly select parameters using these exposed development fixtures.

## Reproduction

- `experiments/logic_first/external_accuracy_tournament.py`
- `.github/workflows/external-accuracy-tournament.yml`
- Full paired cases and aggregate metrics in GitHub Actions artifact.
