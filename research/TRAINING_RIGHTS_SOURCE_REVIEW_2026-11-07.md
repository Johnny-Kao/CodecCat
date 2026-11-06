# P0 source-rights review — evidence and unresolved decisions (2026-11-07)

**Engineering classification only, not legal advice or rights clearance.**
The maintainer confirms that official model training occurred **in Japan** and
accepts residual provenance risk using a transparent citation and rights-inquiry
approach. This is a project risk decision, **not** universal legal clearance.
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

## Japanese statutory basis and accepted residual risk

Japan Copyright Act [Article 30-4](https://laws.e-gov.go.jp/law/345AC0000000048)
permits certain non-enjoyment uses, expressly including information/data analysis
in item (ii), to the necessary extent, subject to the exception for uses
unreasonably prejudicing copyright holders. The
[Agency for Cultural Affairs](https://www.bunka.go.jp/english/policy/copyright/amendments_2018/)
explains the provision in the context of AI training. See also its
[2024 overview](https://www.bunka.go.jp/english/policy/copyright/pdf/94055801_01.pdf)
and [copyright guidance](https://www.bunka.go.jp/seisaku/chosakuken/aiandcopyright.html).

For CodecCat's stated Japan-based encoding-feature analysis, this provides a
plausible statutory basis independent of individual source licenses.
However, it does not itself resolve exceptional source restrictions, the
scope of necessary use, contractual/data-protection issues, or questions
about international model distribution. Historical per-file source licenses
remain unverified as facts even where this statutory basis may apply.

**Documented maintainer decision:** accept residual risk, publish original
sources and official statutory references, preserve the rights inquiry
route, and do not assert absolute rights clearance. This decision does not
require individually negotiated license grants for every one of the 3,086
files as a blanket prerequisite. No automated rights-clearance field is
changed by this document.

## Decision gates

1. Keep current published package history intact. Do not silently replace or
   unpublish artifacts, reinterpret past distribution as legally cleared, or
   change the default model as part of this inventory.
2. For a new model, maintain source citations and record the Japan-based
   information-analysis purpose plus any specific source restrictions.
   Do not automatically demand a separate negotiated license from all
   publishers when a statutory copyright exception may apply.
3. If a specific collection shows an applicable restriction, demonstrable
   infringement risk, or conflict with the statutory exception, investigate
   or exclude that source before preparing a replacement corpus.
4. Re-train any replacement only with approved data, then re-run the frozen
   training replay, independent holdout and paired accuracy/speed/RAM gates.
   Source removal alone does not prove a replacement model is accurate.
5. The maintainer accepts residual project risk; legal authorization is
   not represented as proven. Escalate a concrete rights complaint or
   substantial distribution concern as it arises.

**Review scope:** Do not create repeated rights-inventory PRs or treat
unverified individual licenses as an automatic stop to unrelated quality
research. Continue to disclose sources and respond to substantiated claims;
no document alone is a legal opinion or a universal permission grant.
