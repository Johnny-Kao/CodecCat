"""External legacy-fixture diagnostic. Uses original pinned repository bytes.

These samples are *not* blind external confirmation; no model is trained/promoted.
"""
import codecs, hashlib, json, subprocess
from pathlib import Path
from collections import Counter, defaultdict

SOURCES = {
 "uchardet": ("BYVoid/uchardet", "test"),
 "charset_normalizer": ("jawah/charset_normalizer", "data"),
}
LABELS={"iso-8859-2":"iso8859-2","windows-1250":"cp1250"}
def run(args):
    roots={name:Path(args.root)/name for name in SOURCES}
    records=[]; duplicates=defaultdict(list)
    for name,root in roots.items():
        paths=sorted((root/"test").rglob("*.txt")) if name=="uchardet" else sorted((root/"data").glob("sample-*.txt"))
        for path in paths:
            relative=path.relative_to(root).as_posix()
            if name=="uchardet":
                label=LABELS.get(path.stem.lower())
                if not label:continue
            else:
                # charset-normalizer corpus samples have no reliable codec ground truth here.
                label=None
            raw=path.read_bytes()
            digest=hashlib.sha256(raw).hexdigest()
            entry={"repo":name,"path":relative,"sha256":digest,"bytes":len(raw),
                   "declared_label":label,"label_verified":False,
                   "status":"development-diagnostic-only"}
            if label:
                try:raw.decode(label,"strict");entry["strict_truth_decode"]=True
                except UnicodeError:entry["strict_truth_decode"]=False
            else:entry["strict_truth_decode"]=None
            duplicates[digest].append(name+":"+relative)
            records.append(entry)
    pairs=[r for r in records if r["declared_label"] and r["strict_truth_decode"]]
    summary={"schema":"codeccat-external-candidate-inventory-v1",
             "status":"not-independent-confirmation",
             "total_candidates":len(records),
             "labeled_strict_valid":len(pairs),
             "labels":dict(Counter(r["declared_label"] for r in pairs)),
             "duplicates":{k:v for k,v in duplicates.items() if len(v)>1},
             "notes":["No training. Filename labels are not verified source ground truth.",
                      "Existing third-party detector fixtures may overlap existing benchmarks.",
                      "No original bytes are uploaded in artifact."]}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (out/"inventory.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True) for x in records)+"\n")
    print("EXTERNAL_CANDIDATE_RESULT_START")
    print(json.dumps(summary,indent=2))
    print("EXTERNAL_CANDIDATE_RESULT_END")
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",required=True)
    p.add_argument("--out",default="research-results/external-candidates")
    run(p.parse_args())
