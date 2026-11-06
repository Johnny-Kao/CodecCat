# P0 source-rights review — evidence and unresolved decisions (2026-11-07)

**Engineering classification only, not legal advice or rights clearance.**
Current training/model distribution authorization stays **UNVERIFIED**.
This document uses the already reconciled 3,086-row ledger; it does not
replace or duplicate `datasets/official/rights_inventory.py`.

## Review priority

| Catalog source | Records | Evidence checked | What remains unresolved |
|---|---:|---|---|
| CulturaX | 1,941 | Dataset card delegates use terms to mC4 and OSCAR; dataset mixes cleaned web-derived materials. | Precise sub-source/revision and applicable licenses for each record, underlying publisher rights, personal data, training and derived numerical model distribution. |
| Common Crawl | 353 | Common Crawl ToU warns crawled content may have separate terms of origin owners and requires respecting their copyright. | Original page rights, any individual permissions and lawful basis for intended training/distribution. |
| chardet | 314 | Upstream `chardet/test-data` README: individual files remain copyright their respective publishers. | Trace each upstream test file to its original publisher/license; the `chardet` provenance label is not permission. |
| unknown | 246 | Catalog has no known publisher. | Actual original source, license, and authorization. Unknown records cannot be marked cleared without additional traceability or a lawful basis. |
| Other catalog sources | 232 | Provenance mapped; not individually rights-reviewed here. | Independent review still required. |

The first four groups account for **2,854/3,086 (92.48%)** records. 
This prioritization does **not** imply the remainder is cleared.

### Primary evidence

- [chardet/test-data README](https://github.com/chardet/test-data): test-data licensing warning; pinned source catalog commit `b0c0d206913ba118ffea2f7d306883e7a4a76db7`.
- [CulturaX official dataset card](https://huggingface.co/datasets/uonlp/CulturaX/blob/main/README.md): terms refer to mC4 and OSCAR, with personal/sensitive content warning.
- [Common Crawl Terms of Use](https://www.commoncrawl.org/terms-of-use): third-party crawled content can have additional terms, and IP/third-party rights must be respected.
- [Existing per-file source ledger](./OFFICIAL_TRAINING_RIGHTS_INVENTORY_2026-11-06.md): exact source counts and hashes.

## Technical checks completed vs. missing legal determination

Completed: source grouping for 3,086 records; deterministic fingerprints;
metadata-only CI; no raw training sample distribution in this review.
Not completed: tracing the original publisher/license for each record;
checking the intended commercial/noncommercial scope and jurisdictions;
legal basis for training and distribution of numeric n-gram weights; data
protection obligations; rights-holder permission if needed.

A model holding only weights, IDF, short n-grams and classifier labels does
not itself prove either lawful or unlawful training/distribution. No
automated script should fill `release_authorized=true` based on the source
label or license name alone.

## Decision gates

1. Keep current published package history intact. Do not silently replace or
   unpublish artifacts, reinterpret past distribution as legally cleared, or
   change the default model as part of this inventory.
2. Before promoting a new official model, document intended countries of
   distribution, use cases (including commercial use), lawful basis per
   source, any restrictions, and a record of legal review/permission.
3. If a source group cannot be cleared to the needed standard, **exclude**
   it from a replacement training corpus; do not treat a license URL as
   universal permission for copied web content.
4. Re-train any replacement only with approved data, then re-run the frozen
   training replay, independent holdout and paired accuracy/speed/RAM gates.
   Source removal alone does not prove a replacement model is accurate.
5. Current automated status is **NOT DETERMINED**; rights and release
   decisions belong to an authorized human reviewer, with legal counsel
   where warranted.

**Stop condition:** Do not create additional engineering PRs merely to
repeat the same rights inventory. Escalate unresolved publisher/licensing
questions as an explicit release decision; start a replacement corpus only
after the proposed authorized sources have been identified.
