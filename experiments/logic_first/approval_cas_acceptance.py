"""Approval token scoped to action/digest/generation; rollback and replay tests."""
import json,os,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from codeccat.approval_registry import approval_token,transition
from codeccat.model_lifecycle import _atomic_json,snapshot,SCHEMA
from hashlib import sha256

def reject(fn,cls=Exception):
 try:fn()
 except cls:return
 raise AssertionError("unauthorized transition succeeded")

def main():
 secret=b"isolated test-only issuer secret" + b"\0"*16
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);(root/"models").mkdir();(root/"candidates").mkdir()
  def fixture(raw):
   digest=sha256(raw).hexdigest()
   (root/"models"/(digest+".zip")).write_bytes(raw)
   _atomic_json(root/"candidates"/(digest+".json"),{"model_sha256":digest,"promotable":True,"gate":{"accepted":True}})
   _atomic_json(root/"candidates"/(digest+".confirmation.json"),{
     "digest":digest,"gate_accepted":True,"independence_certified":True,"provenance_reviewed":True})
   return digest
  a=fixture(b"synthetic A");b=fixture(b"synthetic B")
  token_a=approval_token(secret,action="activate",digest=a,expected_generation=0,actor="synthetic-approver")
  reject(lambda:transition(root,action="activate",digest=a,token=token_a,secret=b"different secret"*4),PermissionError)
  assert snapshot(root)["generation"]==0
  transition(root,action="activate",digest=a,token=token_a,secret=secret)
  reject(lambda:transition(root,action="activate",digest=a,token=token_a,secret=secret),PermissionError)
  token_b=approval_token(secret,action="activate",digest=b,expected_generation=1,actor="synthetic-approver")
  reject(lambda:transition(root,action="activate",digest=a,token=token_b,secret=secret),PermissionError)
  transition(root,action="activate",digest=b,token=token_b,secret=secret)
  rollback=approval_token(secret,action="rollback",digest=a,expected_generation=2,actor="synthetic-approver")
  reject(lambda:transition(root,action="rollback",digest=b,token=rollback,secret=secret),ValueError)
  old=root/"models"/(a+".zip");old.write_bytes(b"tampered")
  reject(lambda:transition(root,action="rollback",digest=a,token=rollback,secret=secret),ValueError)
  assert snapshot(root)["active"]==b
  old.write_bytes(b"synthetic A")
  transition(root,action="rollback",digest=a,token=rollback,secret=secret)
  reject(lambda:transition(root,action="rollback",digest=a,token=rollback,secret=secret),ValueError)
  assert snapshot(root)=={"schema":SCHEMA,"active":a,"previous":b,"generation":3}
  print("APPROVAL_CAS_RESULT",json.dumps({"status":"PASS","generation":3,
        "replay_denied":True,"wrong_secret_denied":True,
        "tamper_denied":True,"rollback":"PASS","simulation_only":True}))
if __name__=="__main__":main()
