"""Research-only, fail-closed model lifecycle with local atomic activation and rollback.

Artifacts are content-addressed portable models. Registry never unpickles.
No network, GitHub release, or PyPI operations.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json,os,tempfile,re
from .model_candidate_gate import validate_candidate

SCHEMA="codeccat-lifecycle-v1"
def _canonical(data):return json.dumps(data,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def _atomic_json(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix=".candidate-",suffix=".tmp")
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(_canonical(record));f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)
def _sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def _load_state(root):
    path=Path(root)/"active.json"
    if not path.exists():return {"schema":SCHEMA,"active":None,"previous":None,"generation":0}
    data=json.loads(path.read_text())
    if data.get("schema")!=SCHEMA or not isinstance(data.get("generation"),int):
        raise ValueError("invalid active state")
    return data
def _verify_model(root,digest):
    if not isinstance(digest,str) or re.fullmatch(r"[a-f0-9]{64}",digest) is None:raise ValueError("invalid model digest")
    file=Path(root)/"models"/(digest+".zip")
    if not file.is_file() or _sha(file)!=digest:raise ValueError("model missing or tampered")
    return file
def snapshot(root):
    state=_load_state(root)
    if state["active"] is not None:_verify_model(root,state["active"])
    return state
def stage(root,model,*,train_rows,heldout_rows,provenance,incumbent=None,
          min_samples=20):
    """Stage only on verified disjointness; return decision and immutable id.

    Provenance is a claim supplied by caller, not a cryptographic attestation of
    where rows came from. Production promotion requires independent review.
    """
    train=list(train_rows);heldout=list(heldout_rows)
    if not isinstance(provenance,dict) or not provenance.get("source") or not provenance.get("license"):
        raise ValueError("source and license provenance are required")
    if not train or not heldout:raise ValueError("empty train/validation")
    train_groups={g for _,_,g in train}
    valid_groups={g for _,_,g in heldout}
    if train_groups&valid_groups:raise ValueError("source group leakage")
    train_hashes={sha256(b).hexdigest() for b,_,_ in train}
    valid_hashes={sha256(b).hexdigest() for b,_,_ in heldout}
    if train_hashes&valid_hashes:raise ValueError("exact duplicate leakage")
    root=Path(root)
    with tempfile.TemporaryDirectory() as d:
        candidate=Path(d)/"candidate.zip";model.save_portable(candidate)
        from .vnext_strong_l2 import TrainableDetector
        restored=TrainableDetector.load_portable(candidate)
        if restored.fingerprint!=model.fingerprint:raise ValueError("fingerprint mismatch")
        if incumbent is None:raise ValueError("incumbent mandatory for candidate acceptance")
        decision=validate_candidate(incumbent,restored,heldout,min_samples=min_samples)
        digest=_sha(candidate)
        record={"schema":SCHEMA,"model_sha256":digest,"training_fingerprint":model.fingerprint,
          "provenance":provenance,"train_count":len(train),"validation_count":len(heldout),
          "train_groups_sha256":sha256(_canonical(sorted(train_groups))).hexdigest(),
          "heldout_groups_sha256":sha256(_canonical(sorted(valid_groups))).hexdigest(),
          "gate":{"accepted":decision.accepted,"reasons":list(decision.reason),"report":decision.report},
          "validation_status":"development-only","promotable":False}
        root.joinpath("candidates").mkdir(parents=True,exist_ok=True)
        _atomic_json(root/"candidates"/(digest+".json"),record)
        if decision.accepted:
            root.joinpath("models").mkdir(parents=True,exist_ok=True)
            dest=root/"models"/(digest+".zip")
            if dest.exists():_verify_model(root,digest)
            else:
                with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False,suffix=".tmp") as tmp:
                    tempname=tmp.name;tmp.write(candidate.read_bytes());tmp.flush();os.fsync(tmp.fileno())
                try:os.replace(tempname,dest)
                finally:
                    if os.path.exists(tempname):os.unlink(tempname)
        return record
def certify_confirmation(root,digest,incumbent,confirmation_rows,*,provenance_reviewed=False,min_samples=20):
    """Diagnostic confirmation, NOT automatic independent certification."""
    root=Path(root)
    record=json.loads((root/"candidates"/(digest+".json")).read_text())
    if record.get("model_sha256")!=digest or not record["gate"]["accepted"]:
        raise PermissionError("candidate failed validation")
    from .vnext_strong_l2 import TrainableDetector
    candidate=TrainableDetector.load_portable(_verify_model(root,digest))
    if candidate.fingerprint!=record["training_fingerprint"]:
        raise ValueError("training fingerprint mismatch")
    rows=list(confirmation_rows)
    if not rows:raise ValueError("empty confirmation")
    decision=validate_candidate(incumbent,candidate,rows,min_samples=min_samples)
    status={"schema":SCHEMA,"digest":digest,"gate_accepted":decision.accepted,
      "reasons":list(decision.reason),"provenance_reviewed":bool(provenance_reviewed),
      "evidence":decision.report,"independence_certified":False}
    _atomic_json(root/"candidates"/(digest+".confirmation.json"),status)
    return status

def activate(root,digest,*,approved=False):
    """Explicit human approval required; development-only candidate cannot activate."""
    if not approved:raise PermissionError("explicit approval required")
    root=Path(root);record=json.loads((root/"candidates"/(digest+".json")).read_text())
    if record.get("model_sha256")!=digest or not record["gate"]["accepted"]:
        raise PermissionError("candidate not validated")
    confirmation_file=root/"candidates"/(digest+".confirmation.json")
    if not confirmation_file.exists():raise PermissionError("missing confirmation evidence")
    confirmation=json.loads(confirmation_file.read_text())
    if confirmation.get("digest")!=digest or not confirmation.get("gate_accepted"):
        raise PermissionError("confirmation not passed")
    if not confirmation.get("independence_certified") or not confirmation.get("provenance_reviewed"):
        raise PermissionError("independent source and provenance review required")
    if not record.get("promotable"):
        raise PermissionError("candidate not independently promotion-approved")
    _verify_model(root,digest)
    current=snapshot(root)
    _atomic_json(root/"active.json",{"schema":SCHEMA,"active":digest,
      "previous":current["active"],"generation":current["generation"]+1})
def rollback(root,*,approved=False):
    if not approved:raise PermissionError("explicit rollback approval required")
    root=Path(root);state=snapshot(root);old=state["previous"]
    if old is None:raise ValueError("no previous model")
    _verify_model(root,old)
    _atomic_json(root/"active.json",{"schema":SCHEMA,"active":old,
      "previous":state["active"],"generation":state["generation"]+1})
