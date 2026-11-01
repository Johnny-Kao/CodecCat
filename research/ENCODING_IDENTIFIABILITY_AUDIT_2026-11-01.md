# CodecCat — encoding identifiability audit (2026-11-01)

Status: **GitHub Actions SUCCESS; development diagnostic only**.
Run: https://github.com/Johnny-Kao/CodecCat/actions/runs/41273326077

Source: pinned chardet/test-data `b0c0d206913ba118ffea2f7d306883e7a4a76db7`, 3,086 examples. This corpus is the training-origin development corpus, **not independent acceptance data**. No weights, stable23 features, main, alpha tag, or PyPI changed.

## Key results

| Pair | Different single-byte mappings (of 256) | Same decoded text, truth A | Same decoded text, truth B |
| --- | ---: | ---: | ---: |
| cp037 / cp500 | 7 | 6/34 | 1/25 |
| iso8859-16 / iso8859-2 | 51 | 1/22 | 5/71 |
| iso8859-16 / cp1250 | 79 | 1/22 | 5/44 |
| iso8859-2 / cp1250 | 47 | 29/71 | 8/44 |
| cp1252 / iso8859-1 | 32 | 30/42 | 74/74 |
| cp1252 / iso8859-15 | 40 | 29/42 | 34/41 |
| cp1252 / hp-roman8 | 127 | 4/42 | 12/44 |

“Same decoded text” means that decoding the provided bytes under the other codec gives the exact same Unicode text; encoding label cannot be identified from this input alone. **Different decoded text does NOT mean the true codec is uniquely identifiable**: both outputs may be plausible, and without external provenance the original intent is not observable.

## Attribution

- CP037/CP500 only differ at 7 single-byte values: `4a 4f 5a 5f b0 ba bb`. 7/59 evaluated examples have **no distinguishing byte at all**; 52/59 carry a differing byte but need semantic/contextual disambiguation. The old residual model shifted correctness from CP500 to CP037, so no rule should be promoted on this evidence.
- CP1252 versus ISO-8859-1: all 74 ISO-8859-1 fixtures and 30/42 CP1252 fixtures decode identically under both codecs. Their encoding labels may be unidentifiable, but decoded-text correctness is preserved on those rows.
- ISO-8859-2 versus CP1250: 37/115 samples are identical-text cases across the two codec interpretations. Judge model changes on decoded text as well as label.
- Low group diversity: hp-roman8 has 44 examples but only 5 source identifiers. Counts alone exaggerate source coverage.

## Decision

Do **not** change the Alpha or promote a classifier. First acquire independent, provenance-recorded source documents with verified original encodings. Stratify by distinguishable byte presence, document group, language, and source. For CP037/CP500, deliberately include natural documents using the seven differing bytes. For Latin pairs, include documents with codec-distinguishing characters and genuine linguistic context. Avoid claiming synthetic re-encoded examples are independent sources.

The next evaluation should pair old/new predictions with strict decoded-text correctness, encoding-label accuracy, coverage/abstention, and per-family regression gates. Keep a final untouched external confirmation set. The existing development corpus may be used for diagnostics but not to make a blind acceptance claim.

## Reproduction

- `experiments/logic_first/encoding_identifiability_audit.py`
- `.github/workflows/encoding-identifiability-audit.yml`
- Run artifact `encoding-identifiability-development-audit`: `summary.json`, `cases.jsonl`

This experiment tests intrinsic byte-map ambiguity and observed corpus equivalence, not improved prediction accuracy. No competitor comparison was run.
