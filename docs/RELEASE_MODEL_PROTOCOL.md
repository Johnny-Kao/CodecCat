# Release Model Protocol

## Why this exists

The canonical 365/418 result comes from pooled 4-fold held-out evaluation. It proves the architecture under cross-validation, but it does not produce one deployable fitted model.

Shipping one of the fold models would be methodologically invalid.

## Release-model sequence

1. Freeze runtime implementation.
2. Freeze the development corpus and all architecture decisions.
3. Acquire an independent evaluation corpus with provenance separated from the development corpus.
4. Define label normalization before looking at model errors.
5. Fit one final model using all eligible development/training data.
6. Export the fitted state into CodecCat's runtime-only model format.
7. Freeze the model artifact and its SHA-256 fingerprint.
8. Evaluate exactly once on the independent holdout for the release candidate.
9. Run charset-normalizer and chardet 7 on exactly the same bytes.
10. Publish both strengths and remaining gaps.

## Independence rules

The independent holdout must not be used to:
- select S3;
- change feature representation;
- add or remove rerank rules;
- choose specialist pairs;
- tune confidence gates;
- tune replacement-rate threshold;
- select model hyperparameters.

If the holdout causes architecture changes, it becomes development data and a new independent holdout is required.

## Release gate

A release candidate must have:
- reproducible package build;
- runtime/model separation;
- deterministic model loading;
- independent-holdout report;
- three-way comparator report;
- no known mismatch between exported model behavior and its validation form.

The release decision should prefer a clear Pareto position over benchmark overfitting.
