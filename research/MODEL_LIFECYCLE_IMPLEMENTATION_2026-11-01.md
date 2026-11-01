# Model lifecycle and manifest retraining execution — 2026-11-01

## Verified GitHub Actions

- Model candidate gate: https://github.com/Johnny-Kao/CodecCat/actions/runs/41312989429 — SUCCESS (1,138 held-out, portable parity, destructive candidate rejected).
- P0 model lifecycle: https://github.com/Johnny-Kao/CodecCat/actions/runs/41314095284 — SUCCESS (SHA-256 immutable candidate, source-group/byte isolation, under-sized gate rejection, unapproved activation blocked, no active model altered).
- P1 manifest retrain: https://github.com/Johnny-Kao/CodecCat/actions/runs/41314292829 — SUCCESS (1,103 training, 776 validation, 1,124 untouched confirmation rows; candidate staged, malformed manifest rejected, no activation).

The first P1 run failed because the preexisting development fixture adapter truncates certain multibyte documents at 4,096 bytes; the strict full-document manifest gate correctly rejected incomplete trailing sequences. The test adapter was corrected to use only complete strict-decodable fixture records. This was a test-fixture mismatch, not a model accuracy regression.

## Implemented research interfaces

- `src/codeccat/model_candidate_gate.py`: fail-closed model comparison including wrong-text and label-family regression checks.
- `src/codeccat/model_lifecycle.py`: candidate content SHA-256, data provenance record, group/digest isolation, portable-model reload, atomic local-state update, activation approval check and rollback interface.
- `experiments/logic_first/retrain_manifest.py`: user-supplied manifest train/validation/confirmation input, candidate training and staging; never auto-activates.
- `experiments/logic_first/external_corpus_gate.py`: existing manifest syntax, license/source fields and split-isolation validation.

## Explicitly NOT completed or authorized

- Independent dataset *authenticity* is not verified by metadata alone.
- The untouched confirmation split has been reserved but has NOT been used for independent final release certification.
- Product activation and actual successful rollback have NOT been end-to-end tested because the development candidate is intentionally `promotable=false`. This prevents pretending unreviewed data is approved. Do not claim P0 rollback is proven.
- The lifecycle implementation is research-only, not yet reviewed as a production-safe concurrent registry. Trainable portable model boundaries need adversarial archive sizing/path checks.
- No public release or change to main, Alpha or PyPI.

## Decision needed before candidate activation policy

Choose an authorization/certification policy for model activation: require (A) independent confirmation provenance and no regression + manual explicit approval (recommended), or (B) locally opted-in activation based only on developer validation without independent confirmation (riskier). Regardless, user-managed local experiments may stage models without activation.
