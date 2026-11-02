"""Stability gate for the current Python inference architecture.
No runtime modifications. Same trained artifact, warm/cold and repeated-input parity.
"""
import hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector

def worker(path, data):
 model=TrainableDetector.load_portable(path)
 return [{"encoding":r.encoding,"candidates":r.candidates,"ambiguous":r.ambiguous}
         for raw in data for r in [model.rank(raw,final=len(raw)<=4096)]]
def main():
 root=Path("research-results/runtime-stability");root.mkdir(parents=True,exist_ok=True)
 rows=corpus(Path("training"))
 train=residual_subset([(b,y,g) for b,y,g,k,sha in rows if k!=0])[0]
 test=[b for b,y,g,k,sha in rows if k==0][:300]
 model=TrainableDetector.fit(train)
 path=root/"model.zip";model.save_portable(path)
 original=worker(path,test)
 hashes=[];elapsed=[]
 for i in range(3):
  t=time.process_time()
  result=worker(path,test)
  elapsed.append(round(time.process_time()-t,4))
  assert result==original,"stateful prediction drift"
  hashes.append(hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest())
 for seed in ("0","1","17"):
  p=subprocess.run([sys.executable,__file__,"--worker",str(path)],input=json.dumps([x.hex() for x in test]),text=True,capture_output=True,check=True,env=dict(os.environ,PYTHONHASHSEED=seed))
  assert json.loads(p.stdout)==original,("fresh process mismatch",seed)
 report={"n":len(test),"repeat_runs":3,"fresh_process_seeds":[0,1,17],"identical":True,
         "cpu_seconds_each":elapsed,"fingerprint":hashes[0],
         "model_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
         "scope":"Architecture parity and warm repeated-run stability, not an accuracy benchmark"}
 (root/"report.json").write_text(json.dumps(report,indent=2))
 print("STABILITY_GATE_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":
 if len(sys.argv)>1 and sys.argv[1]=="--worker":
  print(json.dumps(worker(sys.argv[2],[bytes.fromhex(x) for x in json.loads(sys.stdin.read())])))
 else: main()
