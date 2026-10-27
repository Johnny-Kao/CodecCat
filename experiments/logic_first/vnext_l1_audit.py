"""L1 structural upper-bound audit; no fit, learned scores or parameters.

A codec is eliminated only when strict decoding proves it impossible for
the tested complete input; incomplete 4096-byte prefixes remain nonfinal.
"""
import argparse,collections,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_multidist import gather,norm,family
from codeccat.vnext_gate import evaluate

def main(a):
    train=corpus(Path(a.train))
    labels=tuple(sorted({y for b,y,g,f,h in train}))
    train_bytes={hashlib.sha256(b).hexdigest() for b,y,g,f,h in train}
    seen=set();cases=[];all_counts=collections.Counter()
    for origin,root in (("uchardet",Path(a.uchardet)),("unicode-suite",Path(a.unicode_suite))):
        for p,truth in gather(root,origin):
            b=p.read_bytes()
            if not b:continue
            # Model consumes <=4096 bytes. Gate must treat truncated input as incomplete.
            raw=b[:4096]; final=len(b)<=4096
            sha=hashlib.sha256(raw).hexdigest()
            if sha in seen or sha in train_bytes:continue
            seen.add(sha)
            if truth not in labels:continue
            out=evaluate(raw,labels,final=final)
            valid=out.possible
            row={"source":origin,"truth":truth,"family":family(truth),"sha":sha,
                 "bytes":len(raw),"final":final,"remaining":len(valid),
                 "eliminated":len(out.impossible),"truth_wrongly_eliminated":truth in out.impossible,
                 "unique_correct":len(valid)==1 and valid[0]==truth,
                 "unique_wrong":len(valid)==1 and valid[0]!=truth,
                 "bom":out.bom.encoding}
            cases.append(row)
            all_counts["n"]+=1
            all_counts["wrongly_eliminated"]+=int(row["truth_wrongly_eliminated"])
            all_counts["unique_correct"]+=int(row["unique_correct"])
            all_counts["unique_wrong"]+=int(row["unique_wrong"])
    if not cases:raise RuntimeError("no external nonduplicate fixture samples")
    metrics={}
    for name,rr in (("all",cases),*((f,[r for r in cases if r["family"]==f])
                                  for f in sorted({r["family"] for r in cases}))):
        counts=collections.Counter(r["remaining"] for r in rr)
        metrics[name]={"n":len(rr),"wrong_elimination":sum(r["truth_wrongly_eliminated"] for r in rr),
            "unique_correct":sum(r["unique_correct"] for r in rr),
            "unique_wrong":sum(r["unique_wrong"] for r in rr),
            "median_remaining":sorted(r["remaining"] for r in rr)[len(rr)//2],
            "min_remaining":min(r["remaining"] for r in rr),
            "max_remaining":max(r["remaining"] for r in rr),
            "median_eliminated":sorted(r["eliminated"] for r in rr)[len(rr)//2]}
    report={"schema":"l1-structural-audit-v1","total_trained_codecs":len(labels),
            "metrics":metrics,"known_test_sources":True,
            "limitation":"These sources have been inspected before. Structural decoder viability is a necessary condition, not proof of intended codec. Audit uses full fixture where <=4096; no training or tuning."}
    path=Path(a.output);path.mkdir(parents=True,exist_ok=True)
    (path/"report.json").write_text(json.dumps(report,indent=2))
    (path/"rows.jsonl").write_text("\n".join(map(json.dumps,cases))+"\n")
    if metrics["all"]["wrong_elimination"]:
        raise AssertionError("L1 falsely eliminated a fixture truth; inspect ground truth/incomplete semantics")
    print("L1_AUDIT_BEGIN");print(json.dumps(report,indent=2));print("L1_AUDIT_END")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True);p.add_argument("--uchardet",required=True)
    p.add_argument("--unicode-suite",required=True)
    p.add_argument("--output",default="research-results/vnext-l1-audit")
    main(p.parse_args())
