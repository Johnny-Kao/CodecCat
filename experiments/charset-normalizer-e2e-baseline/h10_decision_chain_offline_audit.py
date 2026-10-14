"""Analyze immutable H7 paired trace without re-collecting or tuning data."""
import argparse,csv,json
from collections import defaultdict,Counter
from pathlib import Path

FIELDS=("raw_top1","hybrid_top1","triad_top1","sig_top1","gb_top1","baseline_top1","full_raw_top1","full_final_top1","final_top1")
def stats(rows):
    n=len(rows)
    out={"n":n,"stages":{}}
    for field in FIELDS:
        applicable=[r for r in rows if r.get(field) not in ("",None,"None")]
        out["stages"][field]={"applicable":len(applicable),"hits":sum(r[field]==r["truth"] for r in applicable)}
    out["raw_top3_coverage"]=sum(r["truth"] in (r["raw_top1"],r["raw_top2"],r["raw_top3"]) for r in rows)
    out["raw_correct_final_wrong"]=sum(r["raw_top1"]==r["truth"] and r["final_top1"]!=r["truth"] for r in rows)
    out["raw_wrong_final_correct"]=sum(r["raw_top1"]!=r["truth"] and r["final_top1"]==r["truth"] for r in rows)
    out["raw_top3_missing_and_final_wrong"]=sum(r["truth"] not in (r["raw_top1"],r["raw_top2"],r["raw_top3"]) and r["final_top1"]!=r["truth"] for r in rows)
    transitions={}
    for previous,next_ in zip(FIELDS[:-1],FIELDS[1:]):
        eligible=[r for r in rows if r.get(previous) not in ("",None,"None") and r.get(next_) not in ("",None,"None")]
        transitions[previous+"__"+next_]={
            "n":len(eligible),"changed":sum(r[previous]!=r[next_] for r in eligible),
            "beneficial":sum(r[previous]!=r["truth"] and r[next_]==r["truth"] for r in eligible),
            "harmful":sum(r[previous]==r["truth"] and r[next_]!=r["truth"] for r in eligible)}
    out["transitions"]=transitions
    return out
def run():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--out",default="research-results/h10");a=p.parse_args()
    with Path(a.input).open(newline="") as f:rows=list(csv.DictReader(f))
    assert len(rows)>=1500 and all(r["truth"] and r["final_top1"] for r in rows)
    by={}
    for dims in (("crawl",),("route",),("family",),("route","family"),("crawl","route"),("crawl","route","family")):
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in dims)].append(r)
        by["+".join(dims)]={"|".join(keys):stats(vals) for keys,vals in sorted(groups.items())}
    output={"source":"H7 immutable artifact 38553419903, not a new holdout","overall":stats(rows),"by":by}
    d=Path(a.out);d.mkdir(parents=True,exist_ok=True)
    (d/"decision-chain-audit.json").write_text(json.dumps(output,indent=2))
    print("DECISION_AUDIT_RESULT_START")
    print(json.dumps({"overall":output["overall"],"by_route":by["route"],"by_family":by["family"],"by_crawl":by["crawl"],"rh_family":{k:v for k,v in by["route+family"].items() if k.startswith("RH|")}},indent=2))
    print("DECISION_AUDIT_RESULT_END")
if __name__=="__main__":run()
