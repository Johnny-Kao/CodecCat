# Local model ownership boundary — CodecCat

The package is a **local encoding detector**. No central account, identity service, admin role, HMAC issuer, or server-side approval is required.

## Public product behavior (target)

1. **First use:** use the bundled default model, with no network call, auto-update or account setup.
2. **Train:** user supplies their own dataset; training produces a candidate model and a validation report. Neither training nor validation changes the model used by applications.
3. **Select:** the user explicitly installs a candidate through the local API. The caller is the model administrator. A passed validation gate, matching model digest/training fingerprint and optimistic generation check prevent accidental corruption or stale writes.
4. **Restore:** the user can explicitly return to bundled default at any time; existing user model files are not deleted.
5. **Rollback:** optionally switch to the previous user-selected model where available.
6. **Boundaries:** package checks well-formed inputs and model integrity, but data rights, training quality, thresholds, model deployment choice and repository/OS permissions belong to the application owner.

**No authentication layer:** A caller that can write the local registry already controls the configuration. This package does not attempt to defend against a hostile operating-system user who has write access to both registry and code. This is deliberate, not a missing feature.

## Research implementation

- `src/codeccat/local_model_manager.py`: `current_model`, `select_candidate`, `restore_default`, `rollback_model`.
- `src/codeccat/model_candidate_gate.py`: development candidate comparison.
- `src/codeccat/model_lifecycle.py`: candidate staging and provenance; existing approval-token/third-party-certification prototype is **not part of the intended public API**.
- Tests: `.github/workflows/local-selection-acceptance.yml` on GitHub Actions only.

## Remaining work before publishing

The above is an isolated prototype, not yet integrated with the package's documented runtime/model loader. Before a release: wire the entry point, confirm bundled model compatibility and source distributions, run package tests across supported Python platforms and document the simple API. A passing test in a research branch is not a released version.

No changes to `main`, Alpha release or PyPI here.
