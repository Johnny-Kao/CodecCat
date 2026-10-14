"""H11: mechanism-level RH/CJK candidate-ceiling audit from immutable H7 rows.

Diagnostic only. No fit, no thresholds, no new data, no production change.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

def tally(rows):
    n=len(rows)
    missing=[r for r in rows if r["truth"] not in (r["raw_top1"],r["raw_top2"],r["raw_top3"])]
    failed=[r for r in rows if r["final_top1"]!=r["truth"]]
    return {
        "n":n,"final_errors":len(failed),
        "raw_top3_missing":len(missing),
        "raw_top3_missing_and_final_error":sum(r["final_top1"]!=r["truth"] for r in missing),
        "truth_distribution":dict(Counter(r["truth"] for r in rows)),
        "missing_truth_distribution":dict(Counter(r["truth"] for r in missing)),
        "missing_predicted_top1":dict(Counter(r["raw_top1"] for r in missing)),
        "final_errors_by_truth":dict(Counter(r["truth"] for r in failed)),
        "final_errors_by_prediction":dict(Counter(r["final_top1"] for r in failed)),
        "final_errors_correct_charset_normalizer":sum(r["charset_normalizer"]==r["truth"] for r in failed),
        "final_errors_correct_chardet7":sum(r["chardet7"]==r["truth"] for r in failed),
        "missing_truth_same_family_as_raw_top1":sum(
            r["truth"] in ("shift-jis","euc-jp","euc-kr","big5","gb18030","gbk","cp949")
            and r["raw_top1"] in ("shift-jis","euc-jp","euc-kr","big5","gb18030","gbk","cp949")
            for r in missing),
        "missing_by_byte_length":{
            label:sum((int(r["byte_len"])<=1024 if label=="<=1KiB"
                  else int(r["byte_len"])<=4096 if label=="<=4KiB"
                  else int(r["byte_len"])<=16384 if label=="<=16KiB"
                  else int(r["byte_len"])>16384) for r in missing)
            for label in ("<=1KiB","<=4KiB","<=16KiB",">16KiB")
        },
        "top_missing_examples":[
            {"hash":r["sha256"],"crawl":r["crawl"],"truth":r["truth"],
             "raw_top3":[r["raw_top1"],r["raw_top2"],r["raw_top3"]],
             "final":r["final_top1"],"chardet7":r["chardet7"],
             "charset_normalizer":r["charset_normalizer"],
             "byte_len":int(r["byte_len"]),"margin":float(r["margin"])}
            for r in missing[:40]
        ]
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--out",default="research-results/h11")
    args=p.parse_args()
    with open(args.input,newline="") as f: rows=list(csv.DictReader(f))
    assert len(rows)==1560
    rh=[r for r in rows if r["route"]=="RH"]
    cjk=[r for r in rh if r["family"]=="cjk"]
    out={
        "study":"H11 RH/CJK raw Top-3 ceiling and confusion attribution",
        "source_run":"38553419903",
        "source_is_exploratory":True,
        "rh":tally(rh),
        "rh_cjk":tally(cjk),
        "rh_cjk_by_crawl":{crawl:tally([r for r in cjk if r["crawl"]==crawl])
                           for crawl in sorted({r["crawl"] for r in cjk})},
    }
    d=Path(args.out);d.mkdir(parents=True,exist_ok=True)
    (d/"mechanisms.json").write_text(json.dumps(out,indent=2))
    brief={k:{kk:vv for kk,vv in v.items() if kk!="top_missing_examples"}
           for k,v in out.items() if isinstance(v,dict) and k!="rh_cjk_by_crawl"}
    print("H11_RESULT_START")
    print(json.dumps({"rh":brief["rh"],"rh_cjk":brief["rh_cjk"],
          "by_crawl":{k:{"n":v["n"],"final_errors":v["final_errors"],"raw_top3_missing":v["raw_top3_missing"]}
                      for k,v in out["rh_cjk_by_crawl"].items()}},indent=2))
    print("H11_RESULT_END")

if __name__=="__main__":main()
