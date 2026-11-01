"""Inspect why selective L3 can reverse correct decoded text; no parameter tuning."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_training_strategy_audit import strict_text
from external_accuracy_tournament import truth_label
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_l3_language import LanguageCoherence
def main():
 p=argparse.ArgumentParser();p.add_argument("--training",required=True);p.add_argument("--external",required=True);a=p.parse_args()
 rows=corpus(Path(a.training));train=residual_subset([(b,y,g) for b,y,g,_,_ in rows])[0]
 l2=TrainableDetector.fit(train);l3=LanguageCoherence.fit(train)
 known={hashlib.sha256(strict_text(b,y).replace("\r\n","\n").replace("\r","\n").lstrip("\ufeff").encode()).hexdigest() for b,y,_,_,_ in rows if strict_text(b,y) is not None}
 errors=[]
 for path in sorted((Path(a.external)/"test").rglob("*")):
  if not path.is_file():continue
  truth=truth_label(path)
  if not truth:continue
  raw=path.read_bytes();ref=strict_text(raw,truth)
  if ref is None:continue
  digest=hashlib.sha256(ref.replace("\r\n","\n").replace("\r","\n").lstrip("\ufeff").encode()).hexdigest()
  if digest in known:continue
  ranked=l2.rank(raw,final=len(raw)<4096)
  decision=l3.rerank(ranked,raw,final=len(raw)<4096)
  prior=strict_text(raw,ranked.encoding)
  after=strict_text(raw,decision.encoding)
  if prior==ref and after!=ref:
   values=[]
   for enc in ranked.candidates[:3]:
    decoded=strict_text(raw,enc)
    values.append({"encoding":enc,"same_text_as_reference":decoded==ref,"language_score":l3.score(decoded) if decoded is not None else None})
   errors.append({"path":path.relative_to(a.external).as_posix(),"truth":truth,"base":ranked.encoding,
                  "after":decision.encoding,"l3_reason":decision.reason,"advantage":decision.language_advantage,
                  "l2_margin":ranked.scores[0]-ranked.scores[1],"candidates":values})
 print("L3_HARMS_START");print(json.dumps(errors,indent=2,ensure_ascii=False));print("L3_HARMS_END")
 assert len(errors)==1, "Expected exactly one known external regression; investigation scope changed"
if __name__=="__main__":main()
