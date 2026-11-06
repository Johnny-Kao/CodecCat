# Official historical training source provenance inventory — 2026-11-06

**Status: 3,086/3,086 source records mapped; legal rights status NOT DETERMINED.**

[Public Actions source inventory #42239986920](https://github.com/Johnny-Kao/CodecCat/actions/runs/42239986920) verified the full ordered historical training manifest against the upstream `chardet/test-data` `CATALOG.md` at fixed commit `b0c0d206913ba118ffea2f7d306883e7a4a76db7`.

| Original upstream catalog source | Records |
|---|---:|
| CulturaX | 1,941 |
| Common Crawl | 353 |
| chardet | 314 |
| unknown | **246** |
| Ude | 96 |
| ENCA | 32 |
| uchardet | 32 |
| charset-normalizer | 17 |
| Chromium | 15 |
| Mozilla | 11 |
| MSDN disc | 11 |
| Web Archive | 11 |
| Contributed | 7 |
| **Total** | **3,086** |

Upstream provenance labels: transcoded **1,958**, wild **790**, suite **222**, historic **112**, extracted **4**. These are **source descriptions**, not licenses or rights determinations.

**Important:** upstream `README.md` says each test file is copyrighted by its respective publisher. The public upstream repository does not provide a universal license for all raw examples or clearly establish the right to train on every item or distribute a model derived from it. In particular, `unknown` source entries cannot be considered cleared merely because the labels or Git commits are recorded.

The metadata-only output includes a per-file ledger with the upstream source, provenance and the exact input SHA256. Every row retains `training_rights_status=UNVERIFIED` and `model_distribution_rights_status=UNVERIFIED`. The CI explicitly prevents a source inventory from setting `release_authorized=true`.

This inventory **does not imply the published model is unlawful**: determining the legal basis for training and distributing derived numerical models requires separate factual and legal review. It does mean the engineering team cannot assert broad rights clearance on the evidence currently collected.

## Release-gate next action

Review the origin and applicable licenses/permissions of the largest source families (CulturaX, Common Crawl, chardet and 246 unknown), decide which may be used under the intended distribution and jurisdiction, and document a rights decision per source. If an unacceptable source cannot be cleared, retraining with an authorized corpus and separately verifying accuracy would be required before promoting a new default model. Until that review is completed, **do not treat the inventory or reproducible training as release authorization**.
