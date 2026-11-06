# Western C1-control counterfactual — fresh-source confirmation protocol (2026-11-07)

This protocol was committed **before scoring either source**. The C1 candidate
reranking rule is frozen from the 363-case *development* diagnostic:
if the published Stable output contains Unicode U+0080–U+009F controls,
iterate existing ranked alternatives, and take the first strict-decodable
alternative containing **zero** C1 controls. Otherwise preserve baseline text.
The alternative candidate is already produced by the default model; the
experiment does not train it, add new classes, or change the installed code.

## Sources

- `GNOME/evince` at `e822030005f22231052667d51398a84327004924`.
- `GNOME/epiphany` at `baf23d30acd0ce758632a99aa99f780f45e1802a`.
- Historical train source `chardet/test-data` at
  `b0c0d206913ba118ffea2f7d306883e7a4a76db7`.
- Published `codeccat==0.1.0`, model SHA256
  `16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b`,
  `chardet==7.7.0`.

All original `po/*.po` files are included if they strict-decode as UTF-8 and
contain a `charset=UTF-8` declaration, sorted by filename. Synthetic Western
recodings use **the identical fixed locale-to-codec mapping and first eight
eligible 1024-character chunks per locale/codec** described in
[overall benchmark protocol](OVERALL_BENCHMARK_PROTOCOL_2026-11-07.md),
with `polib==1.2.0`. Do not change selection rules, thresholds or models
after inspecting results. Both source groups remain separate.

Before any scoring, validate the 3,086 training-file hashes and screen each
selected file/chunk for exact whole bytes, exact first-4096-byte prefix, and
>=3 overlapping sampled 128-byte blocks. Fail closed on any flagged sample.
A negative screen cannot prove semantic independence.

## Decision gate

Report baseline and counterfactual exact Unicode correctness, helped/harmed,
silent wrong nonempty output, changed-output count, and by-codec/source
breakdowns. Also include natural UTF-8 regression controls. A legitimate
improvement must show **zero new harms** across both external sources and
nontrivial repair to Western recodings, without increasing existing runtime
dependencies. No source data or decoded text may leave GitHub Actions.

All Latin samples are synthetic recodings of one current GNOME document
family per project; two new projects do not establish population accuracy or
remove language/translation correlations. A clean counterfactual is **not**
permission to ship a Runtime change: separate production API, security,
compatibility and cross-platform testing is required first.
