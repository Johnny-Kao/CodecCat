# CodecCat overall stratified benchmark — preregistered protocol (2026-11-07)

The selection and evaluation criteria in this document precede scoring. This is
**release-accuracy and resource evaluation**, not training, model tuning or PyPI publication.

## Frozen source and package versions

- Fresh source family: `GNOME/gnome-terminal`, revision
  `d11cd622f07760e0281841d6b084a0d9e62484a6`, all `po/*.po` files,
  sorted by relative path; each file belongs to a distinct source document.
- Pre-examined historical confirmation source: CPython
  `f64e51563a85667446793ad61ea521d005ceb171`,
  `Lib/test/cjkencodings/`, all 15 original/UTF-8 pairs (30 files).
- Training source for contamination screening: `chardet/test-data`
  `b0c0d206913ba118ffea2f7d306883e7a4a76db7`,
  manifest SHA256 `8ec3007dbd59336cee6018cfa44aeded39cca205e93f5fc9f24ee5ecf87f3ca9`.
- Installed, released `codeccat==0.1.0` (model SHA256
  `16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b`),
  `chardet==7.7.0` and `charset-normalizer==3.5.2`.

## Four non-pooled cohorts

1. **Natural UTF-8**: every complete gnome-terminal `.po` file that is strict
   UTF-8 and contains a UTF-8 charset header. No length trimming or selecting
   files after model output.
2. **Western single-byte synthetic recodings**: source text from gettext
   `msgstr` translations in the same pinned project. Fixed locales
   `fr,de,es,it,pt,nl,sv,da,fi` re-encoded with `cp1252` and
   `iso8859_15`; `pl,cs,hu` with `cp1250`; `ru,bg` with
   `cp1251`; `el` with `iso8859_7`; `tr` with `cp1254`.
3. **CJK synthetic recodings**: `ja` with `shift_jis,euc_jp,iso2022_jp`;
   `ko` with `cp949,euc_kr`; `zh_CN` with `gbk,gb18030`;
   `zh_TW` with `big5,cp950`.
4. **Real legacy CJK confirmation**: the 15 CPython original files with
   strict decoded Unicode matched to their 15 UTF-8 reference files. This
   cohort was examined previously; no blind performance claim is allowed.

For synthetic cohorts, parse gettext messages using pinned `polib==1.2.0`,
join nonempty translated messages in original order, partition into fixed
1024-character chunks, include the first at most **8** chunks per
locale/codec which are at least 128 characters, contain at least 5 non-ASCII
characters, and can be strictly encoded. Record excluded chunks with counts,
but never substitute after viewing model outputs. Synthetic recoding is a
stress test, **not** evidence of naturally occurring encoding prevalence.
Do not pool multiple recodings of the same text as independent documents.

## Separation, metrics and safety

Run the historical train overlap screen **before scoring**: exact entire
byte SHA256, first-4096-byte SHA256 and >=3 matching sampled 128-byte blocks.
Fail on overlap indicators in new-source cohorts; pre-examined CPython must
also satisfy the same screen. Negative results do not prove absence of
semantic duplicates or transformed text.

Per-cohort counts include correct decoded Unicode (primary), exact normalized
codec labels (secondary), abstention, false nonempty text and document-group
count. A strict decode of a detector's guessed encoding must equal the
independently derived reference text; replacement decoding is incorrect.
Collect per-detector elapsed p50/p95, cumulative CPU and Linux peak RSS in
**separate processes**. These are one-run CI observations, not robust
cross-platform performance guarantees. Raw test files and translated text are
never uploaded. Source manifests and aggregated counts only.

No single universal accuracy number is justified unless a target deployment
population's codec distribution and source weights are specified. Provide
**macro-average of the four cohort accuracy rates only as a descriptive
balanced score**, not as population accuracy. Flag any missing or too-small
category instead of hiding it.

Frozen scope: no default-model change, no Runtime changes, no PyPI release.
