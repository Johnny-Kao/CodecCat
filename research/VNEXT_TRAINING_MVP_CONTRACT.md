# vNext unified training MVP

Status: staging-only, first usable training contract. Not release-ready, not an R21 replacement.

## Purpose
Allow any user to supply a labelled byte corpus and produce a schema-2 bundle loadable by CodecCat's existing Python Runtime. A single command trains fixed HMT768 baseline linear scorers for the four routes; no hand-written Japanese override is required to complete this training cycle.

## Manifest
UTF-8 JSONL with entries such as:

{"path":"files/train-a.bin","encoding":"shift-jis","split":"train","group":"source-a"}
{"path":"files/validation-b.bin","encoding":"shift-jis","split":"validation","group":"source-b"}

Paths resolve inside the manifest directory only. Encodings are normalized *by the dataset author* for this MVP: use CodecCat's canonical labels such as shift-jis, euc-jp, utf-8, utf-8-sig. Train/validation group IDs must not overlap. Identical byte content across splits and conflicting labels are rejected. Both splits must exist and all validation encodings must appear in training.

The file bytes—not the declared label—are the inference input. A dataset's label accuracy is still the operator's responsibility; this MVP does not claim semantic proof of ground truth. Grouping by domain/source is encouraged.

## Command (currently a Python module, not an installed CLI)
Install the optional training dependencies: scikit-learn, scipy, and numpy.

python -m codeccat.training --dataset /path/to/manifest.jsonl --output /path/to/candidate.npz --report /path/to/validation.json --seed 2026

The exported NPZ can be consumed with:

from codeccat import Detector, load_bundle
detector = Detector(load_bundle("/path/to/candidate.npz"))
print(detector.detect(raw_bytes).encoding)

## Design limitations
- **This is a baseline-only reproducible trainer**, not the full R21 ensemble and not Feature Schema v2.
- It uses the existing 518-dimensional HMT768 feature schema, so no new Japanese structural evidence has yet been made trainable.
- A route with fewer than two labels uses a constant route model (one label) or a global fallback (zero labels); class coverage should be improved before real usage.
- Training is deterministic given the pinned training environment, corpus bytes, dataset ordering and seed. Floating-point bitwise identical NPZ files across architectures are not guaranteed.
- The training process has a separate, reproducible validation split but no untouched official acceptance set or automated competitor comparison yet.
- The model loader includes no feature-schema metadata validation. A future bundle schema revision must record feature version and reproducibility provenance.
- No release/PyPI/main changes without sign-off.

## CI
GitHub Actions workflow: .github/workflows/vnext-training-mvp.yml
Tests validate training, model bundle roundtrip, independent source groups, and unsafe path rejection.

## Next steps
1. Complete CI sign-off for the MVP contract.
2. Establish a versioned feature-schema and optional structural evidence extractor with fit/transform parity between offline training and runtime.
3. Replace hard-coded downstream decision branches with a parameterized decision scorer where supported; preserve deterministic BOM semantics.
4. Provide dataset QC, independent benchmark fixture formats, official regression gates, and a stable CLI.
5. Do not conflate a user-trained customized model with a superior official model; publish only after an untouched cross-domain benchmark.

## Optional dependencies and CI separation
Install the trainer with `pip install -e '.[train]'`. The NumPy-only inference package deliberately does not require scikit-learn. Core packaging compatibility may skip the optional trainer contract, while consolidated training CI always installs training dependencies and runs it.
