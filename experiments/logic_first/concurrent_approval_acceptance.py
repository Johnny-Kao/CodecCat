"""Process-level contention and atomic state crash-boundary simulation (CI-only)."""
import json,os,sys,tempfile,multiprocessing as mp
from pathlib import Path
from hashlib import sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from codeccat.approval_registry import approval_token,transition
from codeccat.model_lifecycle import _atomic_json,snapshot

SECRET=b"isolated-local-test-secret-32-bytes" + b"x"*8

def worker(root,token,digest,start,out):
 start.wait()
 try:
  data=transition(root,action="activate",digest=digest,token=token,secret=SECRET)
  out.put({"ok":True,"generation":data["generation"]})
 except Exception as e:out.put({"ok":False,"type":type(e).__name__})
def main():
 ctx=mp.get_context("fork")
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);(root/"models").mkdir();(root/"candidates").mkdir()
  digests=[]
  for raw in (b"same-generation-A",b"same-generation-B"):
   digest=sha256(raw).hexdigest();digests.append(digest)
   (root/"models"/(digest+".zip")).write_bytes(raw)
   _atomic_json(root/"candidates"/(digest+".json"),{"model_sha256":digest,"promotable":True,"gate":{"accepted":True}})
   _atomic_json(root/"candidates"/(digest+".confirmation.json"),{"digest":digest,"gate_accepted":True,"independence_certified":True,"provenance_reviewed":True})
  q=ctx.Queue();start=ctx.Event()
  ps=[ctx.Process(target=worker,args=(str(root),approval_token(SECRET,action="activate",digest=d,expected_generation=0,actor="test-issuer"),d,start,q)) for d in digests]
  for p in ps:p.start()
  start.set()
  result=[q.get(timeout=15) for _ in ps]
  for p in ps:p.join(timeout=15);assert p.exitcode==0
  state=snapshot(root)
  assert sum(x["ok"] for x in result)==1,result
  assert state["generation"]==1 and state["active"] in digests
  # A partial temporary state record must never become active on its own.
  (root/".candidate-interrupted.tmp").write_text('{"active":"invalid"')
  assert snapshot(root)==state
  # Tampering with active model must fail closed (no silent fallback).
  active=root/"models"/(state["active"]+".zip");active.write_bytes(b"corruption")
  try:snapshot(root)
  except ValueError:pass
  else:raise AssertionError("tampered active model accepted")
  print("CONCURRENCY_RESULT",json.dumps({"status":"PASS","contenders":2,
    "successful_transitions":1,"stale_generation_rejected":True,
    "interrupted_temp_ignored":True,"corrupt_active_rejected":True,
    "crash_durability_full_system_verified":False}))
if __name__=="__main__":main()
