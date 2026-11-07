# Experimental opt-in encoding ambiguity diagnostic (Staging)

**Status:** proposed Staging functionality; not present in the published v0.1.0 PyPI Wheel.

### Goal

Expose a narrow, reproducible sign of multiple possible Unicode interpretations **without modifying the returned decoding**. The default `detect()` and `decode()` methods, published model, ranking, `Detection.ambiguous`, `reason`, candidates, BOM behavior and optional L3 behavior remain unchanged.

### Example (Staging Wheel only)

```python
from codeccat import detect
from codeccat.diagnostics import inspect_ambiguity

raw = b"Price: \x80; total."
detected = detect(raw)
warning = inspect_ambiguity(raw, detected)

if warning.warning:
    print(warning.reason, warning.alternative_encodings)
    # Request a trusted codec from the user or inspect file metadata.
```

A warning is emitted only when:
- The original result contains a Unicode C1 control (U+0080..U+009F);
- There is no recognized BOM;
- The selected text exactly reproduces from the original bytes under the selected codec;
- One of the **already returned** alternative codec labels strictly decodes the same original bytes to **different Unicode without C1 controls**.

The result is the frozen `AmbiguityDiagnostic(warning: bool, reason: str, alternative_encodings: tuple[str, ...])`. This is a *supplemental* diagnostic. It never substitutes the alternative text and never converts its warning to `Detection.ambiguous=True`.

### Explicit limitations

- Legitimate ISO-8859-1 C1 controls may be intentional. **A warning is evidence of ambiguity, not a verified misclassification.**
- Failing to warn does not establish correctness; candidates are limited to the model's current top three. The diagnostic is **not calibrated confidence**.
- No additional ML dependencies. Candidate strict decoding happens only on opted-in calls whose decoded text contains C1.
- Callers must pass the **same bytes** used for detection; the implementation rejects inconsistencies by declining to warn. It cannot prove identity if both interpretations happen to match.
- This candidate does not alter the public 0.1.x contract and requires a separate measured performance and expanded source test before a future release. The published 0.1.0 Wheel is untouched.

### Test expectations

The Staging CI builds a new native Wheel from the existing frozen model and exercises all existing stable model, public API, packaging, install and reliability tests, plus dedicated legitimate C1/BOM/L3/alternative/bytes-like/result immutability tests. Failure blocks advancement. It does not automatically publish or merge a Runtime feature.
