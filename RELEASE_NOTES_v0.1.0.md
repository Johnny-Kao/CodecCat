# CodecCat v0.1.0 — First Stable Release

CodecCat is a native Python character-encoding detector and strict Unicode decoder. This release formalizes its **0.1.x API compatibility contract** and improves distribution portability without retraining the model or changing the proven `v0.1.0a5` decoder logic.

## Stable behavior and compatibility

- Supports Python **3.10–3.14** on released Linux, macOS, and Windows targets.
- Public `CodecCat`, `detect`, `decode`, `Detection` fields, BOM policy and CLI flags have a documented compatibility contract: [docs/STABILITY.md](https://github.com/Johnny-Kao/CodecCat/blob/main/docs/STABILITY.md).
- Invalid decoding is not silently replaced. Encoding inference itself cannot guarantee unique correctness for arbitrary bytes.
- Native inference has no NumPy/scikit-learn dependencies; legacy API compatibility remains available as an optional extra.

## Portability and release engineering

- Genuine **manylinux_2_28** Linux Wheels for x86-64 and ARM64, built and tested in corresponding policy containers.
- macOS x86-64/ARM64 and Windows x86-64/ARM64 Wheels tested on their own GitHub-hosted runners.
- Frozen default model SHA256 is identical across six Wheels. Every asset has a checksum in `SHA256SUMS` and metadata in `MODEL_PROVENANCE.json`.
- Wheel build uses verified PEP-517 packaging to include the model files; avoids the previous bug where a direct maturin build omitted its bundled model.

## Accuracy evidence and limitations

The decoder model and inference algorithm are unchanged from `v0.1.0a5`. That version scored **826/843 (97.98%)** correctly decoded text on a fixed engineering benchmark, compared with chardet's **829/843 (98.34%)**. The tested 843 examples **include development and training sources**, so this is not an independently representative real-world accuracy rate.

Codec identification of sufficiently ambiguous bytes cannot be universally perfect. The inference model was derived from third-party text with mixed upstream rights; [rights and provenance](https://github.com/Johnny-Kao/CodecCat/blob/main/RIGHTS_AND_PROVENANCE.md) remain documented and are not an assertion of legal clearance.

**Install**: select the Wheel for your OS and CPU from the assets below. On supported Linux/Python versions, the `manylinux_2_28` Wheels are preferable to the earlier plain `linux_*` Alpha files. The release is distributed on GitHub; a PyPI upload is a separate action and not claimed here.
