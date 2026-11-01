"""Exercise baseline parity, portable reload, and destructive regression rejection on Actions."""
import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector,Ranked
from codeccat.model_candidate_gate import validate_candidate

class BrokenCandidate:
 def __init__(self,real):self.fingerprint=real.fingerprint+"-broken"
 def rank(self,data,*,final=True):return Ranked(None,(),(),None,True)

def main():
 rows=corpus(Path("training"))
 train=[(b,y,g) for b,y,g,f,_ in rows if f!=0]
 held=[(b,y,g) for b,y,g,f,_ in rows if f==0]
 assert {g for _,_,g in train}.isdisjoint({g for _,_,g in held})
 model=TrainableDetector.fit(residual_subset(train)[0])
 with tempfile.TemporaryDirectory() as d:
  f=Path(d)/"candidate.zip";model.save_portable(f)
  restored=TrainableDetector.load_portable(f)
  pass_decision=validate_candidate(model,restored,held)
  fail_decision=validate_candidate(model,BrokenCandidate(model),held)
  sparse=validate_candidate(model,restored,held[:1])
 assert pass_decision.accepted,pass_decision
 assert not fail_decision.accepted and "aggregate-regression" in fail_decision.reason,fail_decision
 assert not sparse.accepted and "insufficient-heldout" in sparse.reason,sparse
 assert pass_decision.report["old_correct"]==pass_decision.report["candidate_correct"]
 report={"schema":"codeccat-model-gate-test-v1","n":len(held),
  "portable_parity":"PASS","regressed_candidate_rejected":"PASS",
  "insufficient_evidence_rejected":"PASS",
  "incumbent_correct":pass_decision.report["old_correct"],
  "bad_candidate_correct":fail_decision.report["candidate_correct"],
  "bad_candidate_reasons":fail_decision.reason,
  "publication_side_effects":"none"}
 print("GATE_RESULT_START");print(json.dumps(report,indent=2));print("GATE_RESULT_END")
if __name__=="__main__":main()
