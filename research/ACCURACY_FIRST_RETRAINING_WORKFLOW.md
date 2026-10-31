# CodecCat — Accuracy-First Retraining Workflow (Research v1)

**Status:** implemented as experimental tooling on `research/training-strategy-audit`. Not an approved public training API.  
**Product position:** CodecCat is a re-trainable encoding-detection framework. The bundled model is a default snapshot, not the core product. Users should be able to bring their own documents, retrain as distributions drift, compare candidate/previous model, and roll back. This requires quality gates, not merely `fit()`.

## Scope and priorities

1. **Accuracy and correctness first:** optimize cross-source **strictly decoded text correctness**, while reporting exact codec-label accuracy separately. Handle unknowable historical codec labels rather than pretending there is always one definitive answer.
2. **Retrainability as a reliable workflow:** model and runtime remain independent; input dataset provenance, original bytes, source group, rights and train/validation assignment are versioned.
3. **Cost only after quality holds:** do not optimize CPU, speed or RAM or add unproven complexity at this stage.
4. **Research cannot affect production:** no automatic writes to `main`, the `v0.1.0a1` Alpha, Releases or PyPI; no work or tests on the owner's Mac. GitHub Actions is the test environment.

## Protocol: ingestion → attribution → training → candidate comparison → external confirmation → possible promotion

| Gate | What actually runs | Failure mode / resulting action |
|---|---|---|
| **0. Provenance** | Immutable source and data hash; self-reported origin and rights for every file; labeled bytes must strictly decode | Missing data, origin/rights, invalid codec or invalid bytes ⇒ **REJECT**. Self-attested rights do not establish legal verification |
| **1. Contamination** | Check duplicate bytes, decoded-document hashes, and source groups across training vs. validation | Any identical document or source group on both sides ⇒ **REJECT**. Near-duplicates still require review |
| **2. Reproducibility** | Fixed feature profile, versions, corpus, seeds; hash **learned vocabulary + IDF + weights + intercept + predictions** in fresh processes | Different learned state or predictions on the same rows ⇒ **BLOCK**, investigate before reporting accuracy deltas |
| **3. Candidate** | Train `stable23` L2 from supplied labeled raw bytes; `full` or existing `residual` sampling policy; save and reload non-executable numeric portable model | Fit, serialization or equivalence failure ⇒ **REJECT** |
| **4. Development holdout** | Evaluate candidate and optional incumbent on *identical* untouched validation bytes; store label correctness, literal decoded text correctness, abstain rate, family breakdown and paired helps/harms | No incumbent ⇒ **RESEARCH ONLY**; negative net or concentrated regression ⇒ revise **training**; don't alter heldout to win |
| **5. Fresh external confirmation** | Lock a genuinely new document-source-disjoint rights-reviewed blind set *before testing*, evaluate frozen incumbent/candidate and both `chardet 7` and `charset-normalizer` on identical inputs | Underperforming fresh source / bad family regression / invalid data ⇒ **NO GO**. Once inspected, blind set becomes dev diagnostics |
| **6. Promotion** | Owner reviews provenance, risk/coverage, new-source paired evidence, versioned bundle and rollback; only then a separately authorized release action | **No automatic model promotion or PyPI** |

**No single overall accuracy number is sufficient.** Preserve case-level traces so improvement on one codec cannot hide regressions on another. In particular, distinguish labels that cannot be identified from bytes: in this corpus, 104/116 `cp1252`/`iso8859-1` examples have the same decoded text under both codecs, and 63/83 `cp1252`/`iso8859-15` do likewise. Choosing the historically correct name without observable evidence is not a legitimate correctness guarantee.

## Already executed / primary findings

- [Training strategy #41230402172](https://github.com/Johnny-Kao/CodecCat/actions/runs/41230402172): `stable23` residual 2,828/3,086 correct text in the original development folds; Full 2,809; naive class-capped 2,789. Do not promote naive rebalancing.
- [Document integrity #41231227499](https://github.com/Johnny-Kao/CodecCat/actions/runs/41231227499): 26 exact-decoded-document groups cross original heldout folds, touching 89/3,086 evaluation rows; 34 decoded documents occur in multiple filename groups. **The prior filename-based split has known document leakage.**
- [Paired removal #41231337547](https://github.com/Johnny-Kao/CodecCat/actions/runs/41231337547): only remove cross-fold identical decoded text from training. Retaining contamination: 2,830 correct / 3,086; content-disjoint: 2,804 / 3,086 (net -26), on the same heldout inputs. **This is a development-fold result, not a proof that removing leakage decreases true generalization.**
- **Unresolved replay issue:** independently executed reference residual fits on apparently identical source revisions yield slightly different aggregate counts (2,828 vs 2,830) and tens of differing label predictions. The learned-parameter replay workflow must be checked before treating small differences as meaningful; the difference is **not** an established model improvement.
- [Custom corpus contract #41231473940](https://github.com/Johnny-Kao/CodecCat/actions/runs/41231473940): end-to-end candidate model, portable roundtrip, baseline comparison and fail-closed negative tests passed.
- `main`, Alpha tag and PyPI untouched during this research.

## How to supply training data

Use `experiments/logic_first/vnext_custom_retrain_gate.py` on a versioned dataset directory with an explicitly assigned `train` or `validation` group.

```json
{"path":"train/example1.bin","encoding":"utf-8","source_group":"document-A","split":"train","origin":"owned-documents","rights":"owned"}
{"path":"validation/example2.bin","encoding":"cp1252","source_group":"document-B","split":"validation","origin":"owned-documents","rights":"owned"}
```

Each record must point to a real file in the specified root; multiple classes, at least two train samples/class and at least one source-disjoint validation group are required. **Sample above shows schema, not a sufficient dataset to train.** The source/rights strings are a self-attested ledger; training on a source does not establish permission to redistribute it or trained weights.

**Research command** (run in GitHub Actions, not locally):

```bash
PYTHONPATH=src python experiments/logic_first/vnext_custom_retrain_gate.py \
  --root dataset --manifest dataset/manifest.jsonl \
  --policy residual --incumbent previous-model.ccnext
```

The gate saves `candidate.ccnext`, `report.json` and `validation.jsonl`. Every report carries `decision=RESEARCH_CANDIDATE_ONLY` and `promotion_allowed=false`; the gate will not silently bless a model when no blind confirmation or legal review exists.

## Next accuracy experiments, with stop conditions

- **P0: reproducibility failure:** Compare actual fitted numeric state across fresh Python processes and fit order. If unstable, stop small-gain model comparisons.
- **P1: content-disjoint fold rebuild:** freeze connected source-group + exact text groups; ensure class coverage; do nested model selection without reusing previously inspected rows as a final blind set.
- **P2: contrastive EBCDIC / Latin samples:** obtain new independently sourced, rights-reviewed real documents; preserve both labels for text-equivalent encodings and record uncertainty; train/evaluate to fix CP037/500 and ISO/Windows confusion *without* favoring one class at the expense of the other. No ad hoc case-specific rules.
- **P3: external acceptance:** current `chardet 7` and `charset-normalizer` versions pinned, same byte corpus and decode semantics; only promote if cross-source wrong-text rate improves without unacceptable family regressions and evidence is reproducible.
- **Performance phase is intentionally deferred.**

## Source of truth

- [Document overlap & codec identifiability](../experiments/logic_first/vnext_document_identity_audit.py)
- [Paired cross-fold overlap removal](../experiments/logic_first/vnext_leakage_removal_audit.py)
- [Custom-data retrain quality gate](../experiments/logic_first/vnext_custom_retrain_gate.py)
- [Custom retrain contract tests](../tests/test_vnext_custom_retrain_gate.py)
- [Fit/learned-state replay](../experiments/logic_first/vnext_fit_determinism_replay.py)

CI success is a claim about *execution and contracts*, never an automatic declaration that the model is ready or outperforms competitors.
