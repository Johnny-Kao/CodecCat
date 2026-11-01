"""P0.3 isolated simulation. Synthetic approval records MUST NOT be used in production."""
import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from codeccat.model_lifecycle import _atomic_json,SCHEMA,_verify_model,snapshot,activate,rollback,certify_confirmation,stage
from codeccat.vnext_strong_l2 import TrainableDetector
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset

def rejected(fn,exc=Exception):
 try:fn()
 except exc:return
 raise AssertionError("unexpected authorization or state transition")

def main():
 rows=corpus(Path("training"))
 train=[(b,y,g) for b,y,g,k,_ in rows if k==0]
 held=[(b,y,g) for b,y,g,k,_ in rows if k==1]
 confirm=[(b,y,g) for b,y,g,k,_ in rows if k==2]
 assert {g for _,_,g in train}.isdisjoint({g for _,_,g in held+confirm})
 model_a=TrainableDetector.fit(train)
 model_b=TrainableDetector.fit(residual_subset(train)[0])
 provenance={"source":"test-fixtures","license":"test-only","revision":"pinned"}
 with tempfile.TemporaryDirectory() as d:
  root=Path(d)
  staged=stage(root,model_a,train_rows=train,heldout_rows=held,provenance=provenance,incumbent=model_a)
  assert staged["gate"]["accepted"]
  a=staged["model_sha256"]
  assert snapshot(root)["active"] is None
  rejected(lambda:activate(root,a,approved=True),PermissionError)
  status=certify_confirmation(root,a,model_a,confirm)
  assert status["gate_accepted"] and not status["independence_certified"]
  rejected(lambda:activate(root,a,approved=True),PermissionError)
  # Test-only certification fixture: explicitly simulate external human review.
  # It is NOT an attestation of the research corpus's independence.
  record=json.loads((root/"candidates"/(a+".json")).read_text())
  record["promotable"]=True
  _atomic_json(root/"candidates"/(a+".json"),record)
  status["independence_certified"]=True;status["provenance_reviewed"]=True
  _atomic_json(root/"candidates"/(a+".confirmation.json"),status)
  rejected(lambda:activate(root,a),PermissionError)
  activate(root,a,approved=True)
  assert snapshot(root)["active"]==a and snapshot(root)["generation"]==1
  # Second distinct valid model: manually serialize a test-only model to registry.
  tmp=root/"second.zip";model_b.save_portable(tmp)
  from codeccat.model_lifecycle import _sha
  b=_sha(tmp)
  assert b!=a
  dest=root/"models"/(b+".zip");dest.write_bytes(tmp.read_bytes())
  second=dict(record);second["model_sha256"]=b;second["training_fingerprint"]=model_b.fingerprint
  _atomic_json(root/"candidates"/(b+".json"),second)
  reviewed=dict(status);reviewed["digest"]=b
  _atomic_json(root/"candidates"/(b+".confirmation.json"),reviewed)
  rejected(lambda:activate(root,b),PermissionError)
  activate(root,b,approved=True)
  assert snapshot(root)["active"]==b and snapshot(root)["previous"]==a and snapshot(root)["generation"]==2
  rejected(lambda:rollback(root),PermissionError)
  # Missing/tampered previous model must stop rollback and retain new active.
  oldfile=root/"models"/(a+".zip");raw=oldfile.read_bytes();oldfile.write_bytes(raw+b"tampered")
  rejected(lambda:rollback(root,approved=True),ValueError)
  assert snapshot(root)["active"]==b
  oldfile.write_bytes(raw)
  rollback(root,approved=True)
  assert snapshot(root)["active"]==a and snapshot(root)["previous"]==b and snapshot(root)["generation"]==3
  print("ROLLBACK_E2E_START")
  print(json.dumps({"schema":"codeccat-p03-simulation-v1",
    "switch_sequence":["none","A","B","A"],"generation":3,
    "explicit_approval_enforced":True,"confirmation_required":True,
    "tampered_previous_blocked":True,"active_preserved_after_failed_rollback":True,
    "simulation_only":True,"independent_provenance_real":False},indent=2))
  print("ROLLBACK_E2E_END")
if __name__=="__main__":main()
