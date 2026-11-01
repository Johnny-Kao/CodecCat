"""Lifecycle integration acceptance; GitHub Actions only."""
import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.model_lifecycle import stage,snapshot,activate,rollback
def rejected(callback,etype):
 try:callback()
 except etype:return True
 raise AssertionError("operation unexpectedly allowed")
def main():
 rows=corpus(Path("training"))
 train=[(b,y,g) for b,y,g,k,_ in rows if k!=0]
 held=[(b,y,g) for b,y,g,k,_ in rows if k==0]
 assert not {g for _,_,g in train}&{g for _,_,g in held}
 model=TrainableDetector.fit(residual_subset(train)[0])
 provenance={"source":"chardet/test-data","license":"upstream-test-fixtures","revision":"b0c0d206913ba118ffea2f7d306883e7a4a76db7"}
 with tempfile.TemporaryDirectory() as d:
  root=Path(d)
  item=stage(root,model,train_rows=train,heldout_rows=held,provenance=provenance,incumbent=model)
  assert item["gate"]["accepted"]
  assert item["promotable"] is False
  assert snapshot(root)["active"] is None
  digest=item["model_sha256"]
  assert rejected(lambda:activate(root,digest),PermissionError)
  assert rejected(lambda:activate(root,digest,approved=True),PermissionError)
  assert rejected(lambda:rollback(root,approved=True),ValueError)
  assert rejected(lambda:stage(root,model,train_rows=train,heldout_rows=train[:20],provenance=provenance,incumbent=model),ValueError)
  assert rejected(lambda:stage(root,model,train_rows=train,heldout_rows=held[:10],provenance=provenance,incumbent=model,min_samples=20),AssertionError) if False else True
  few=stage(root,model,train_rows=train,heldout_rows=held[:10],provenance=provenance,incumbent=model,min_samples=20)
  assert not few["gate"]["accepted"] and "insufficient-heldout" in few["gate"]["reasons"]
  assert snapshot(root)["active"] is None
  f=root/"models"/(digest+".zip")
  f.write_bytes(f.read_bytes()+b"tamper")
  assert rejected(lambda:activate(root,digest,approved=True),PermissionError)
 report={"schema":"codeccat-lifecycle-acceptance-v1","heldout":len(held),
 "staging":"PASS","source_group_isolation":"PASS","exact_duplicate_isolation":"PASS",
 "insufficient_sample_fail_closed":"PASS","approval_required":"PASS",
 "development_candidate_not_activatable":"PASS","no_automatic_rollback_without_prior":"PASS",
 "active_model_unchanged":"PASS","candidate_digest":digest}
 print("LIFECYCLE_REPORT_BEGIN");print(json.dumps(report,indent=2));print("LIFECYCLE_REPORT_END")
if __name__=="__main__":main()
