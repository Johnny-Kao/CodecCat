# Arbitrary-byte and model steady-state acceptance — 2026-11-02

## CI evidence
- Initial [input robustness run #41377525572](https://github.com/Johnny-Kao/CodecCat/actions/runs/41377525572): PASS for released Stable23.
- Extended [both models run #41377659360](https://github.com/Johnny-Kao/CodecCat/actions/runs/41377659360): PASS for Stable23 and improved Binary TF SGD alpha=0.0003.
- 197 deterministic cases, including empty, ASCII, invalid UTF sequences, partial BOMs, NUL-dense inputs, high-entropy deterministic bytes, lengths crossing 4096 prefix boundary, and 128 KiB long cases.
- For both models: repeated predictions match restored portable-model predictions. Both accept 196/197 and abstain for 1 case. Neither crashes in tested paths.
- Earlier [cross-process steady-state run #41370960064](https://github.com/Johnny-Kao/CodecCat/actions/runs/41370960064) tested Stable23 on 300 held-out documents and hash seeds 0/1/17. **The new binary model has NOT yet been subjected to that same cross-process hash-seed test.**

## Important unresolved risk
Successful strict codec decoding **does not prove actual text**. High-entropy random bytes are often accepted by permissive code pages. The 196/197 rate is a strong signal not to claim robust *semantic* detection of arbitrary binary inputs. Current guarantees are only stability/no unexpected exception for these tested cases, not unlimited input size, arbitrary user types, arbitrary input content or cross-platform stability.

## Engineering implications
1. Keep structural contract and semantic accept/abstain policy distinct. Consider a binary-input detection policy; do NOT tune it to this small generated set alone.
2. Test fresh-process memory, high-volume calls, extreme size handling, and binary-model cross-process reproducibility.
3. Do not change v0.1.0a2 or silently deploy experimental alpha=0.0003.
4. Accuracy development corpus best currently 2897/3086=93.88%; chardet7.5.0 remains ahead on previous benchmark. No independent large-source proof of accuracy improvements.
