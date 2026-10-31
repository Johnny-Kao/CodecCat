# CodecCat

CodecCat is a Python character-encoding detector and strict text decoder.

**Development branch:** `staging/initial-package`  
**Current usable artifact:** model-bearing `0.1.0a1` wheel from the [CodecCat usable alpha wheel GitHub Actions workflow](https://github.com/Johnny-Kao/CodecCat/actions/workflows/alpha-usable-wheel.yml).  
**Release status:** internal alpha; not on PyPI. `main` is unchanged.

## Use it now

Download the **`codeccat-usable-alpha-wheel-and-model-provenance`** artifact from a successful GitHub Actions run, extract the wheel, and install it:

```bash
python -m pip install ./codeccat-0.1.0a1-py3-none-any.whl
```

The **model-bearing alpha wheel** includes a trained `stable23` L2 detector and an optional L3 language-coherence model. An ordinary install from this source branch does **not** include the model snapshot until the distribution/provenance review is complete.

Python API:

```python
import codeccat

data = "日本語を含む UTF-8 の文章".encode("utf-8")

result = codeccat.detect(data)
print(result.encoding, result.text, result.ambiguous)

text = codeccat.decode(data)  # str | None; no lossy replacement
```

Command-line API:

```bash
codeccat document.txt
codeccat --json document.txt
codeccat --decode document.txt > decoded.txt
cat document.txt | python -m codeccat --json
```

Inspect `Detection.accepted`, `Detection.ambiguous`, and `Detection.candidates` if you need to distinguish text correctness from certainty about the original codec name. `bom_policy="strip"` is the default. Input is a complete byte sequence, not a streaming API. The CLI rejects files over 16 MiB rather than silently truncating them.

For a custom portable numeric model instead of the package default:

```python
from codeccat import CodecCat

detector = CodecCat.from_files("my-stable23.zip")
result = detector.detect(b"raw bytes")
```

L3 can be enabled explicitly with `CodecCat(use_l3=True)`; it is off by default because source-independent net benefit is not established and activation increases inference latency.

## Scope and evidence

- L1 validates structural impossibility and handles strong BOM/UTF/escape evidence.
- L2 is a trained byte-level n-gram detector with strict candidate decodes.
- Optional L3 compares decoded Unicode text coherence for uncertain top-two candidates.
- The trained snapshot is built reproducibly in GitHub Actions from a pinned corpus checkout and checked through an actual clean wheel installation and command-line smoke test.
- External results differ across datasets. On 34 synthetic cases from the independent `DereckLee/ecd` repository, stable23 decoded 25 correctly versus 23 for chardet 7.6; the previously evaluated 197-case ICU fixture favored chardet. **Do not infer overall superiority from either sample.**

## Distribution limitation

The current development training source is `chardet/test-data`, whose README states that individual test files remain copyrighted by their respective publishers. Before placing trained model weights in a public source checkout, release, or PyPI package, their provenance and downstream redistribution risk require review. For now, the model-bearing wheel is an **internal testing artifact**, not a published production release.

The repository also retains historical legacy `Detector(RuntimeBundle)` APIs; they are separate from the alpha `CodecCat` facade.

Technical details: [vNext final review gate](research/VNEXT_FINAL_REVIEW_GATE.md), [release model protocol](docs/RELEASE_MODEL_PROTOCOL.md), and [project plan](research/PROJECT_PLAN.md).
