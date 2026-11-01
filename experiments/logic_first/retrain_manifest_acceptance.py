"""End-to-end retrain manifest test; generated documents are fixture derivatives, never independent evidence."""
import json,sys,tempfile,codecs
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from codeccat.vnext_strong_l2 import TrainableDetector
from retrain_manifest import execute

def main():
 rows=corpus(Path("training"))
 bygroup={}
 for raw,label,group,fold,digest in rows:
  try: raw.decode(label,errors='strict')
  except UnicodeError: continue  # 4096-byte development prefixes can end mid-character
  bygroup.setdefault(group,[]).append((raw,label,fold,digest))
 groups={split:[] for split in ("train","validation","confirmation")}
 for group,entries in sorted(bygroup.items()):
  fold=entries[0][2]
  groups[("train","validation","confirmation")[fold]].append((group,entries))
 assert all(groups.values())
 with tempfile.TemporaryDirectory() as td:
  root=Path(td)/"corpus";root.mkdir()
  lines=[];samples=[]
  for split,items in groups.items():
   for group,entries in items:
    for i,(raw,label,fold,digest) in enumerate(entries):
     name=split+"/"+digest+".bin"
     path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
     lines.append({"path":name,"encoding":label,"source_id":group,
         "document_id":group+"/"+digest,"split":split,"license":"test-fixture-only",
         "origin_url":"https://github.com/chardet/test-data",
         "verification_method":"source_metadata"})
     if split=="train":samples.append((raw,label,group))
  manifest=root/"manifest.jsonl"
  manifest.write_text("\n".join(json.dumps(x) for x in lines)+"\n")
  incumbent=TrainableDetector.fit(samples)
  incumbent_file=Path(td)/"incumbent.zip";incumbent.save_portable(incumbent_file)
  out=Path(td)/"candidate"
  report=execute(manifest,root,incumbent_file,out)
  assert report["gate"]["accepted"],report["gate"]
  assert report["active_changed"] is False and report["release_ready"] is False
  assert len(list((out/"models").glob("*.zip")))==1
  # Invalid validation isolation is rejected before any model training.
  mutated=list(lines)
  orig=next(x for x in mutated if x["split"]=="validation")
  duplicate=dict(orig);duplicate["path"]=orig["path"];duplicate["split"]="train"
  bad=root/"bad.jsonl";bad.write_text("\n".join(json.dumps(x) for x in mutated+[duplicate])+"\n")
  try:execute(bad,root,incumbent_file,Path(td)/"bad-candidate")
  except ValueError:pass
  else:raise AssertionError("invalid manifest accepted")
  print("RETRAIN_ACCEPTANCE_START")
  print(json.dumps({"schema":"codeccat-retrain-acceptance-v1","status":"PASS",
        "train":len(samples),"validation":report["gate"]["report"]["n"],
        "confirmation_untouched":report["confirmation_count_untouched"],
        "invalid_manifest_rejected":True,"model_staged":True,
        "active_changed":False},indent=2))
  print("RETRAIN_ACCEPTANCE_END")
if __name__=="__main__":main()
