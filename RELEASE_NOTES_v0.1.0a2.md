# CodecCat v0.1.0a2 — Alpha

Experimental source-backed GitHub wheel release; **not a PyPI publication**.

## Included
- Stable23 remains the default bundled encoding model, with strict decoding and BOM handling.
- Installable model-bearing wheel, Python `CodecCat` / `detect` / `decode` API, and `codeccat` CLI.
- Optional L3 remains off by default. Models are numeric-only portable archives.
- No claims of superior accuracy to chardet or charset-normalizer.

## Continuing research, not shipped
- Binary TF (1–2 gram) showed 2,882/3,086 correct in competitor-owned development fixtures versus Stable23 2,828/3,086, with class-level regressions.
- Independent external 20-fixture sample: Stable23 19/20, Binary TF 19/20.
- The experimental binary-format serializer and trial weights remain in the **research branch**, not this release.

## Release criteria
- Python 3.10–3.13 install-and-run matrix and source unit tests.
- Wheel includes `stable23.zip`, L3 JSON and provenance manifest.
- Fresh-venv API + CLI smoke passes.
- GitHub prerelease published only for matching `v0.1.0a2` tag, after release checks.

## Post-release
Continue source-diverse accuracy study; maintain backwards-compatible API and explicit opt-in model selection.
