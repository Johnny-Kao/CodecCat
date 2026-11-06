# C1 guard adversarial safety and uncertainty research — 2026-11-07

**Decision: reject unconditional C1 reranking as a default Runtime change.** Preserve published CodecCat 0.1.0. This study deliberately tries to falsify the apparent zero-regression result on GNOME synthetic Western text.

## Adversarial Latin-1 source

[Adversarial CI #42272293592](https://github.com/Johnny-Kao/CodecCat/actions/runs/42272293592) PASS, using exact published 0.1.0 frozen model and original deterministic Latin-1 strings. Inputs combine four text templates, three lengths, eleven genuine byte controls in the ISO-8859-1 C1 range, and three insertion positions (4×3×11×3 = **396** deliberately non-representative counterexamples). No source text exported.

| Metric | Unmodified Stable | C1 rerank candidate |
|---|---:|---:|
| Correct Unicode | **99/396** | **0/396** |
| Correct Stable samples newly corrupted | — | **99** |
| Candidate helped | — | 0 |

This refutes an unconditional rule that presence of Unicode C1 controls means the first encoding is erroneous. In a valid Latin-1 text, C1 controls may be the intended literal codepoints. The exact same byte stream may also admit another strict decode with printable Windows-1252 punctuation. Without surrounding metadata or a trusted source encoding, a decoder cannot universally distinguish the intentions.

Previously inspected/derived evidence remains useful but cannot override this failure: original Western benchmark 135→167/178; preregistered Evince/Epiphany synthetic Western 228→318/339, without observed harms on those samples; original uchardet 95→97/102. Those samples **lack the adversarial contract case** and are structurally correlated synthetic recodings. Their apparent 0-harm result does not imply safe default deployment.

## Non-destructive alternative: expose uncertainty

[Four-cohort screening CI #42272532120](https://github.com/Johnny-Kao/CodecCat/actions/runs/42272532120) PASS. A diagnostic warning fires only when the selected decoded text contains C1 controls **and** an already-ranked candidate can strict-decode the same bytes without any C1 controls. It never mutates the original encoding, text, candidate order, BOM or language-model output. On the existing 363-case diagnostic corpus:

| Cohort | Inputs | Wrong decodes | Warned wrong | Warned correct | Wrong remaining unflagged |
|---|---:|---:|---:|---:|---:|
| Natural UTF-8 | 100 | 0 | 0 | 0 | 0 |
| Western synthetic | 178 | 43 | **32** | 0 | 11 |
| CJK synthetic | 70 | 0 | 0 | 0 | 0 |
| Previously examined original CPython CJK | 15 | 4 | **4** | 0 | 0 |

The 396 deliberately legal Latin-1 adversarial samples show this warning must **not** be interpreted as proof of incorrectness; among the 99 baseline-correct adversarial cases the unconditional reranker corrupted every one. An uncertainty flag would preserve their text, but might still create additional warnings. The zero flagged-correct count in the 363-case dataset is not a population precision estimate.

**Compatibility note:** production `Detection.ambiguous` and `Detection.reason` are public result semantics. Changing them silently may be an API contract change. Prefer a separately named, explicit **opt-in** diagnostic (e.g. documented suspicious-control warning), or a staged proposal with tests and compatibility approval. Merely reusing `ambiguous=True` without reviewing upstream contracts is not acceptable.

## Remaining improvement opportunities

1. **Risk-first:** opt-in warnings for "strict text can be decoded in multiple materially different ways"; do not silently replace decoded text. Include why an output was considered suspicious, with candidate evidence but not fictional probabilities.
2. **Higher accuracy:** investigate actual model scores and missing Latin-encoding class coverage using source-disjoint real legacy documents. Existing three-candidate limitation may prevent correct candidate selection; widening the set raises decoding CPU and API consistency costs.
3. **Performance:** a warning can be computed only for suspicious cases (C1 present); skip extra decoding on normal UTF-8/CJK paths. Measure p50, p95, CPU and RSS across separate target architectures before shipping.
4. **Verification:** test allowed C1 controls, BOM, `bom_policy`, opt-in L3, no replacement characters, ambiguity flags and `reason` invariants. Protect the API's stability rather than assuming semantics.

**No runtime/model/API/PyPI change has been made.** Tests are metadata-only and use historical samples as engineering diagnostics, not general real-world accuracy guarantees.
