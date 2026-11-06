# Western Legacy C1-control investigation — 2026-11-07

**Status: research-only; no change to released CodecCat 0.1.0.**

## Root cause evidence

[Failure-diagnosis CI #42268534214](https://github.com/Johnny-Kao/CodecCat/actions/runs/42268534214) reproduced all 43 erroneous nonempty Western outputs from [the 363-case overall benchmark](OVERALL_BENCHMARK_RESULTS_2026-11-07.md). The correct source encoding was sometimes present in the model's top three candidates, but the published detector selected an incorrect, still strictly decodable codec and marked some failures `ambiguous=False`. This is a **ranking/acceptance problem**, not a broken strict Unicode decoder.

| True codec | Wrong count | Dominant wrong predicted codecs |
|---|---:|---|
| cp1252 | 29/72 | iso8859-1 (14), iso8859-15 (9), hp-roman8 (4), cp865 (1), iso8859-10 (1) |
| cp1250 | 5/24 | iso8859-2 (5) |
| cp1254 | 5/8 | iso8859-9 (5) |
| iso8859-15 | 4/50 | hp-roman8 (3), cp865 (1) |

Across all **178** Western synthetic inputs, **zero** strict-decode successfully as UTF-8. Lowering the existing UTF-8 rescue threshold cannot fix these observed mistakes. Indeed, [previous adversarial research](STABLE_FAILURE_DECISION_2026-11-07.md) showed lowering that threshold can harm previously correct Windows-1252 cases.

## Controlled hypothesis: C1-control guard

A possible **conservative reranker** identifies an emitted decoded text containing Unicode C1 controls `U+0080..U+009F`. If one of the *already ranked* model candidates strictly decodes the exact same bytes without C1 controls, it chooses the first such alternative; otherwise returns the original result. The experiment modifies only an in-memory counterfactual output, **not the model, API, artifact or Runtime**.

[Paired 363-case CI #42268865899](https://github.com/Johnny-Kao/CodecCat/actions/runs/42268865899) PASS:

| Development cohort | Published baseline | C1 rerank | Helped | Harmed |
|---|---:|---:|---:|---:|
| Natural UTF-8 (100) | 100 | 100 | 0 | 0 |
| Western synthetic (178) | 135 | **167** | **32** | **0** |
| CJK synthetic (70) | 70 | 70 | 0 | 0 |
| CPython natural CJK (15) | 11 | 11 | 0 | 0 |

Specifically cp1252 improves **43/72 → 65/72**, with zero observed harms on that development set. Western silent misdecodes decrease **43 → 11**. A separate rejection (abstention) counterfactual kept the 135 correct Western outputs while reducing silent wrong text **43 → 8**, but abstained on 35 of 178 inputs; thus it trades successful text recovery for uncertainty and is **not** proposed as a default API change.

## Fresh-source confirmation — selection frozen BEFORE scoring

Preregistered [two-source protocol](WESTERN_C1_FRESH_SOURCE_PROTOCOL_2026-11-07.md) uses GNOME/evince and GNOME/epiphany pinned commits and the exact same already-fixed guard/locale/codec/chunk selection. [Fresh CI #42269257334](https://github.com/Johnny-Kao/CodecCat/actions/runs/42269257334) PASS; the entire historical-training byte/prefix/sampled-block overlap screen was negative before scoring:

| New source and cohort | N | Published baseline | C1 rerank | Helped | Harmed |
|---|---:|---:|---:|---:|---:|
| evince, natural UTF-8 | 101 | 101 | 101 | 0 | 0 |
| evince, Western synthetic | 153 | 96 | **146** | **50** | **0** |
| epiphany, natural UTF-8 | 110 | 109 | 109 | 0 | 0 |
| epiphany, Western synthetic | 186 | 132 | **172** | **40** | **0** |
| **Fresh Western total** | **339** | **228/339 (67.26%)** | **318/339 (93.81%)** | **90** | **0** |

Western silent errors: fresh total **111 → 21**. Natural UTF-8: **210/211** before/after, unchanged. All external sources were downloaded ephemerally and only metadata and aggregated results uploaded.

**Interpretation:** A principled C1-character anomaly explains a substantial share of the previously observed Western misclassifications, and this exact rule replicated its benefit across two additional source projects with no observed regressions. This is stronger than a one-file fix or retuning to the 29 original errors.

## Original Legacy bytes regression (previously inspected source)

[Original-file regression CI #42269753746](https://github.com/Johnny-Kao/CodecCat/actions/runs/42269753746) PASS on pinned original `BYVoid/uchardet` test bytes (the established 102-case extraction): **95/102 → 97/102**, helped 2, harmed 0; the candidate changed 3 outputs. This source is **not blind or clean of training overlap**, so it only checks that the C1 hypothesis is not limited to synthetically recoded PO text. It also provides no guarantee against legitimate C1 characters or other unseen Legacy source families.

## Why this is not ready for main Runtime

- A C1 control character **may be intentionally present** in genuine ISO-8859-1 text. Reranking could corrupt intended text in cases not represented by these samples. Negative results on 728+ samples cannot establish universal zero-harm.
- Both new Western pools were **synthetically recoded from UTF-8 translation catalogs**, not naturally collected CP1252 production files. They share language, structure and chunk selection patterns; differences between two projects do not make samples independent draws from the full real-world population.
- The production implementation must preserve explicit BOM handling, the `bom_policy` contract, optional L3 behavior, candidate ordering, `ambiguous` and `reason` semantics, and low-latency/zero-ML-dependency requirements. The counterfactual only alters output text; it does **not** yet implement consistent metadata behavior.
- Candidate reranking also needs adversarial counterexamples, production-origin Western byte files, full published-wheel regression suite, six-platform build/install/behavior tests, and paired performance/RSS checks.

**Decision:** keep shipped 0.1.0 unchanged. The C1 guard is an evidence-backed candidate for a **separate staged implementation and review**, not yet a release-ready patch. It should be implemented at the decode decision layer after ranking, not by retraining the model or lowering the UTF-8 rescue threshold.

Raw source bytes, translated catalog content, model weights, and original Unicode strings were never uploaded as test artifacts.
