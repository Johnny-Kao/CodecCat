# Selective L3 failure root-cause and architecture decision — 2026-11-01

Actions: https://github.com/Johnny-Kao/CodecCat/actions/runs/41311241271 (SUCCESS).

## Reproduced harmful case

Original `uchardet/test/zh/big5.txt`, labeled Big5. Stable23 residual L2 selects **big5** and decodes the reference text correctly. Selective L3 reranks to **cp273** (incorrect text). Its existing multilingual Unicode trigram model assigns `big5` -16.1937509 vs `cp273` -15.5605205; `cp273` receives the larger score by +0.6332304, above the original L3 threshold. L2 margin is 0.0816417. Third candidate cp037 receives the same language score as cp273. This is a failure in the premise that higher global text trigram likelihood is sufficient proof to override an encoding decision.

## Decision

- Do not fix with a per-codec exception or threshold search.
- Keep L3 opt-in and disabled by default in alpha/main.
- Reframe the design: L1 only proves byte/structural impossibility or strong signatures; L2 is trainable ranker; optional textual evidence cannot override an otherwise structurally defensible candidate solely based on a global mixed-script trigram score.
- Future alternative should make the language verifier **non-authoritative by default**, with explicit provenance and script-coherence evidence, and provide abstention when the evidence cannot discriminate. It must be verified on unrelated original documents and regressions, alongside speed/RAM costs.
- Next engineering work should target reliable structural evidence where mathematically justified and a robust retrain/validation/rollback user workflow, not repeated tuning on these same fixtures.
- This run confirms case-specific failure; it does not demonstrate a repaired model.

Main, alpha, PyPI untouched. No unverified claims of broad improvement.
