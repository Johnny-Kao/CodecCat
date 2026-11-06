"""P0 acceptance: a fresh public Runtime must consume a model trained offline.

Run from an environment containing only the published codeccat runtime,
not NumPy, scikit-learn or codeccat-train. It does not assert production
accuracy for these small synthetic fixtures.
"""
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path

import codeccat

root=Path("native-prototype/training/smoke-data")
model=Path("trainer-results/custom.ccm")
report=json.loads(Path("trainer-results/training-report.json").read_text())
manifest=root/"manifest.jsonl"

assert md.version("codeccat")=="0.1.0",md.version("codeccat")
for name in ("numpy","scipy","sklearn","codeccat_train"):
    assert importlib.util.find_spec(name) is None, f"runtime depends on trainer: {name}"

assert model.read_bytes()[:4] == b"CCM1"
model_sha=hashlib.sha256(model.read_bytes()).hexdigest()
assert model_sha==report["model_sha256"]
assert report["schema"]=="codeccat-offline-train-v1"
assert report["training_n"]>=40 and report["validation_n"]>=10
assert set(report["classes"])=={"cp1252","utf-8"}

custom=codeccat.CodecCat.from_files(model)
assert custom.model_fingerprint
counts={"n":0,"accepted":0,"text_exact":0,"label_exact":0}
by_codec={}
for line in manifest.read_text(encoding="utf-8").splitlines():
    rec=json.loads(line)
    if rec["split"]!="validation":
        continue
    raw=(root/rec["path"]).read_bytes()
    truth=raw.decode(rec["encoding"],errors="strict")
    result=custom.detect(raw)
    repeat=custom.detect(raw)
    assert result==repeat,"same model/bytes non-determinism"
    assert isinstance(result.candidates,tuple)
    assert isinstance(result.ambiguous,bool)
    assert result.text is None or isinstance(result.text,str)
    d=by_codec.setdefault(rec["encoding"],{"n":0,"accepted":0,"text_exact":0,"label_exact":0})
    counts["n"]+=1;d["n"]+=1
    for target in (counts,d):
        target["accepted"]+=int(result.accepted)
        target["text_exact"]+=int(result.text==truth)
        target["label_exact"]+=int(result.encoding==rec["encoding"])
assert counts["n"]==12,counts
assert counts["accepted"]>0,"trained model cannot decode any validation fixture"
assert all(v["accepted"]>0 for v in by_codec.values()),by_codec
assert all(v["text_exact"]>0 for v in by_codec.values()),by_codec
print("TRAINER_NATIVE_PUBLISHED_RUNTIME_E2E_PASS",
      json.dumps({"model_sha256":model_sha,"counts":counts,"by_codec":by_codec},
                 sort_keys=True),flush=True)
