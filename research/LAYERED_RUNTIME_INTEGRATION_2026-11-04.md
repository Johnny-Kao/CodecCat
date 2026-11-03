# Layered runtime — integrated acceptance gate (2026-11-04)

## What actually ran

**Research-only executable code:** `src/codeccat/vnext_layered_pipeline.py`.
- Semantic binary preflight (reject high-confidence nontext, abstain instead of pretending all decodable bytes are text).
- Existing deterministic BOM / escape / Unicode checks and fixed Binary TF L2 (alpha=0.0003, 1–2-grams).
- Explicit XML, HTML, CSS and Python charset declaration evidence takes precedence when declared encoding strictly decodes the full bytes.
- Unicode trigram L3 is applied only when no document declaration was recognized (fixed development-selected margin<=0.5 and advantage>0.15).
- Strict complete-document decode, optional BOM strip; produces a structured detection, not an unguarded label.

**Code remains opt-in on `research/encoding-identifiability-audit`. Released `v0.1.0a2` and `main` unchanged.**

## Gates and evidence

- Original paired architectural components: https://github.com/Johnny-Kao/CodecCat/actions/runs/41667718873 (PASS).
- Semantic abstention text coverage and synthetic negatives: https://github.com/Johnny-Kao/CodecCat/actions/runs/41670396270 (PASS).
- Integrated strict decoding, three external sources and authentic Enca binary: https://github.com/Johnny-Kao/CodecCat/actions/runs/41670474086 (PASS).

### Accuracy: never mix these denominators

1. **Historical prefix/final=False development scoring:** 2,944/3,086 = 95.40%. This is a *source-group-fold development score* in which a 4,096-byte fixture can be decoded with final=False (allowing incomplete byte sequences or shifts). The declaration-only stage gives 2,931/3,086, original L2 2,897/3,086. This 95.40% is NOT a complete-document runtime measurement.
2. **Strict full-document runtime:** 3,003 of the 3,086 raw fixture bytes decode successfully with their declared true codec using final=True. The integrated pipeline accepted all 3,003 and returned exact Unicode content for **2,858/3,003 = 95.17%** (full BOM handling). There are 83 true-codec streams not valid as complete documents under strict decoding; 1,570 fixtures have exactly 4,096 bytes (possible truncation). Do not count invalid-as-complete streams as valid complete-text samples.
3. **External strict full-document sources:** Enca **45/81**, uchardet **19/19**, charset-normalizer **7/8**. The layered runtime preserved baseline outcomes: **no net external gain**. These fixtures remain small, source-correlated and filename-labeled; near-duplicate underlying text cannot be excluded by byte hashes alone.
4. **Negative acceptance:** 73/98 synthetic negatives rejected, 25 accepted. The real Enca `test/binary` fixture (five repeated 0xFF bytes) is rejected by a general repetitive-high-byte rule; this is a single authentic binary example and no demonstration of generalized binary coverage. The 3,086 development positives and 81 Enca text fixtures were accepted by the standalone preflight; strict-valid complete-text runtime rejection = 0/3,003.

### Runtime cost

Integrated `detect()` with training models precomputed: median process CPU 2.407 ms/document, p95 5.253 ms/document on Ubuntu CI fixture mix. Exact hardware, fixture length and environment affect timings. Numbers exclude model fitting and are not a matched comparison with other projects.

### Remaining problems and promotion blockers

- **No independent external accuracy gain** demonstrated. The 95% threshold is development-scoped; L3 thresholds were selected on the same development source.
- Six original development correct cases were broken in the layered offline experiment; examples include UTF-16 overridden by wrong legacy codepages, and metadata declarations disagreeing with the true byte encoding. These tradeoffs are unresolved.
- 28 of 36 errors in Enca originally missed the truth in Top3. Cheap byte family candidates improve candidate recall (53/81 -> 70/81), but a reliable selector is still missing. Naive centroid override performed worse and must not ship.
- Semantic binary rejection remains incomplete on real data; do not treat a single real fixture as representative.
- The integrated runtime itself needs reproducibility/portable model/parity tests in fresh processes before production use.

## Decision

Keep the composed research runtime and its tests as a **candidate**, not a release. Continue with (1) source-provenance-verified and independent held-out encodings, (2) a calibrated family-level selector whose regressions are explicitly tracked, and (3) real-format binary/malformed input gates. Avoid further threshold tuning on the Enca or the same chardet development fixture; no C port until correctness settles.
