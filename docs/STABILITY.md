# Public API compatibility contract

This document defines the intended compatibility guarantees for the first non-prerelease CodecCat version. Accuracy is a measured property, **not a promise that every unknown byte stream has an objectively unique encoding**.

## Scope

- Python **3.10–3.14**, with native wheels for supported Linux, macOS and Windows architectures.
- `import codeccat`, `codeccat.CodecCat`, `codeccat.detect`, `codeccat.decode` and the `Detection` result.
- `detect(data, bom_policy="strip")` accepts `bytes`, `bytearray`, `memoryview` and returns a `Detection`. The method `CodecCat.detect` has the same input/result contract.
- `decode(...)` returns a `str` or `None`, never silently decoding with `errors="replace"`.
- The immutable `Detection` result keeps: `encoding`, `text`, `candidates`, `bom_encoding`, `bom_present`, `ambiguous`, `reason`, `bom_policy`, `l3_called`, `l3_changed`, and the derived `accepted` property.
- `bom_policy="strip"` removes the signature character when appropriate; `"preserve"` keeps it. Invalid `bom_policy` raises `ValueError`.
- Non-bytes-like inputs raise `TypeError`. Unknown/undecodable data can be returned as `Detection(encoding=None, text=None,...)`.
- The CLI `codeccat FILE`, `--decode`, `--json`, `--bom-policy` and standard input are supported. It emits UTF-8 bytes for decoded output, exits nonzero if it cannot decode, and refuses inputs over 16 MiB rather than truncating.
- Optional legacy interface `codeccat[legacy]` exists for compatible prior model-loader methods. Legacy functionality requires NumPy; default native inference does not.

## Version policy

For the **0.1.x stable line**: patch releases preserve public method names, input types, result fields, and successful CLI flag names. Internal ranking, candidate scores, the trained model and heuristic outcomes may change, and must pass regression gates. Exact encoding detection for arbitrary bytes is **not** guaranteed, nor are specific `reason` values, candidate ordering, or model fingerprints stable across releases.

Breaking public interface changes require a separately announced version line and migration notes. Prior-to-0.1.0 Alpha releases are experimental and not covered by this commitment.

## Boundaries

This is a **whole-buffer** decoder; no streaming API or incremental text protocol is promised. Invalid bytes are not replaced, but a syntactically valid prediction can still yield incorrect Unicode text. If an encoding label is security-sensitive or authoritative, supply a known codec rather than inferring it. The reported `ambiguous` flag is a warning, not a calibrated posterior confidence.

The model comes from third-party training text, whose legal provenance is described in [RIGHTS_AND_PROVENANCE.md](../RIGHTS_AND_PROVENANCE.md). This agreement does not assert legal clearance or coverage of every deployment environment.
