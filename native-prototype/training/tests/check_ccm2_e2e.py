"""Check the shared trainer's compact CCM2 model in published Stable Runtime.

This synthetic fixture validates portability and model format, not accuracy.
"""
import hashlib
import json
from pathlib import Path

from codeccat import CodecCat

model = Path("trainer-results/custom-f32.ccm")
report = json.loads(Path("trainer-results/training-report-f32.json").read_text())
data = model.read_bytes()
assert data[:4] == b"CCM2"
assert report["model_schema"] == "CCM2"
assert report["weight_precision"] == "f32"
assert report["model_sha256"] == hashlib.sha256(data).hexdigest()

f64 = Path("trainer-results/custom.ccm")
assert f64.read_bytes()[:4] == b"CCM1"
assert len(data) < f64.stat().st_size, "f32 export should be smaller"

custom = CodecCat.from_files(model)
assert custom.model_fingerprint
root = Path("native-prototype/training/smoke-data")
manifest = root / "manifest.jsonl"
accepted = 0
for line in manifest.read_text(encoding="utf-8").splitlines():
    rec = json.loads(line)
    if rec["split"] != "validation":
        continue
    raw = (root / rec["path"]).read_bytes()
    result = custom.detect(raw)
    assert result == custom.detect(raw), "model output is not deterministic"
    assert result.text is None or isinstance(result.text, str)
    accepted += int(result.accepted)
assert accepted > 0, "compact model cannot decode any validation fixture"
print("TRAINER_CCM2_PUBLISHED_RUNTIME_PASS", report["model_sha256"], accepted)
