# CodecCat external corpus acceptance contract — 2026-11-01

Status: **Implemented provenance contract with synthetic contract tests; no external acceptance corpus yet.**

Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41276798068

## Objective

The old 3,086-example `chardet/test-data` corpus has already been used repeatedly for development. It must not be treated as an untouched external confirmation set. Before a model-promotion claim, new real-world documents need reproducible provenance and isolation of source and document identities.

## Input

Provide a JSONL manifest and a root directory containing the real source bytes. One row per file, for example:

```json
{"path":"samples/example.bin","encoding":"cp1250","source_id":"archive-A","document_id":"document-001","split":"confirmation","license":"CC0-1.0","origin_url":"https://example.org/original","verification_method":"source_metadata"}
```

The URL above is a **format example**, not an identified dataset or proof of licensing. Accepted splits are `train`, `validation`, `confirmation`. Verification methods are `source_metadata`, `creator_attestation`, and `verified_original`. None automatically certifies authenticity.

## Command (GitHub Actions / isolated Linux)

```sh
python experiments/logic_first/external_corpus_gate.py --manifest corpus.jsonl --root external-corpus --out research-results/external-corpus-gate/report.json
```

The gate checks required fields, codec support, strict full-document decoding, safe file paths, exact byte duplicates across splits, and source/document isolation across splits. It rejects missing confirmation data. The CLI exits nonzero on failures.

## Known limitations

- **Not a real data acquisition result:** no independent corpus was downloaded or evaluated.
- Provenance, original encoding, license and right to redistribute still require human/manual evidence review. An entered license is **not verified** by this tool.
- Hash equality does not detect different byte encodings of the same text, near-duplicates, translations, boilerplate, or the same upstream original under different document IDs.
- A source group string is only as trustworthy as its assignment. All documents in a logical original-source family must use the same source_id.
- The current gate deliberately has no model training, ranking or benchmark code. Use it to quarantine and document real data first.
- Never print raw user documents or automatically publish unlicensed corpora through GitHub Actions artifacts.

## Promotion path

1. Identify original document sources and secure reuse rights; retain the evidence.
2. Lock source-level partitions and reserve a never-inspected confirmation split.
3. Verify duplicates and semantic near-duplicates across splits before training.
4. Train alternatives only on train, choose on validation, and check the confirmation split exactly once.
5. Compare paired same-text accuracy, label accuracy and abstention/coverage; report per-encoding-family regressions.
6. Promote only if predeclared gates pass, with a reproducible model fingerprint and reversible previous-model fallback.

No changes to main, alpha, PyPI, or production model.
