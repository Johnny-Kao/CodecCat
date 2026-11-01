# P0.3 explicit approval and rollback simulation — 2026-11-01

## Final Actions results

- https://github.com/Johnny-Kao/CodecCat/actions/runs/41314979148 — **PASS** for explicit approval and simulated A → B → A rollback.
- https://github.com/Johnny-Kao/CodecCat/actions/runs/41314979173 — **PASS** for lifecycle registry consistency and weak-evidence overwrite prevention.
- Earlier passing simulation: https://github.com/Johnny-Kao/CodecCat/actions/runs/41314748183.

## Demonstrated

1. Candidate with no confirmation record cannot be activated.
2. Candidate with confirmation marked not independently certified cannot be activated.
3. Passing a programmatic approval flag is insufficient while confirmation and promotion gates remain false.
4. Under a **simulated** approved/certified record written only inside the isolated test directory, transitions none → A → B → A succeed, generations 1, 2, 3.
5. Unapproved activation or rollback is blocked.
6. A tampered previous model is rejected during rollback; the current active model is preserved.
7. Candidate training fingerprint collision with changed gate evidence is rejected rather than overwriting the original candidate record, including cases where portable ZIP bytes have different hashes.
8. No changes to main, public release, PyPI, or any user's active model.

## Important product limitations

- This is a **simulation**, not independently certified external provenance or actual user approval of a concrete model. The test changes JSON files in a temporary test registry to imitate a decision.
- `approved=True` is not an authenticated user-identity or signed approval. Real deployment should use a trusted approval mechanism, immutable audit trail, generation-based compare-and-swap and process synchronization/locking.
- The research confirmation method currently records `independence_certified=False` by design. It cannot certify an unverified third-party corpus.
- Promotion remains off until externally reviewed confirmation material exists and explicit user approval is captured.
- No concurrency, interrupted write/OS crash, cross-process transaction, corrupted metadata adversarial audit or release-package integration has passed.
- The lifecycle registry is research-only. Model files are hash-checked; this alone does not authenticate their producer.

## Next safe direction

Prepare a real independent confirmation dataset, independently verify provenance and source groups, harden the approval/authentication boundary, and test concurrent/failed transactions in GitHub Actions before any staging promotion.
