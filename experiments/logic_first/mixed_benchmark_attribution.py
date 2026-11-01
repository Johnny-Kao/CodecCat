"""Error attribution for the full benchmark; no model tuning."""
import json
from collections import defaultdict,Counter
from pathlib import Path

root=Path("research-results/mixed-random-benchmark")
models={}
for key in ("codeccat","chardet","charset-normalizer"):
    models[key]={r["sha"]:r for r in json.loads((root/(key+".json")).read_text())["cases"]}
keys=set(models["codeccat"])
assert keys==set(models["chardet"])==set(models["charset-normalizer"])
family=defaultdict(lambda:Counter())
paired=Counter()
losses=[];wins=[]
for sha in sorted(keys):
    cc=models["codeccat"][sha];ch=models["chardet"][sha];cn=models["charset-normalizer"][sha]
    assert cc["truth"]==ch["truth"]==cn["truth"] and cc["source"]==ch["source"]==cn["source"]
    label=cc["truth"]
    f=family[label]
    f["n"]+=1
    for name,row in (("cc",cc),("ch",ch),("cn",cn)):
        f[name+"_correct"]+=int(row["correct"])
    if ch["correct"] and not cc["correct"]:
        f["ch_only"]+=1;paired["ch_only"]+=1
        losses.append({"sha":sha,"truth":label,"source":cc["source"],"cc_guess":cc["prediction"],"ch_guess":ch["prediction"],"cn_guess":cn["prediction"],"bytes":cc["bytes"]})
    if cc["correct"] and not ch["correct"]:
        f["cc_only"]+=1;paired["cc_only"]+=1
        wins.append({"sha":sha,"truth":label,"source":cc["source"],"cc_guess":cc["prediction"],"ch_guess":ch["prediction"],"cn_guess":cn["prediction"],"bytes":cc["bytes"]})
    if not cc["correct"] and not ch["correct"]:paired["both_wrong"]+=1
    if cc["correct"] and ch["correct"]:paired["both_right"]+=1
summary=[{"truth":k,**dict(v),"net_chardet_advantage":v["ch_only"]-v["cc_only"]} for k,v in family.items()]
summary.sort(key=lambda x:(-x["net_chardet_advantage"],-x["ch_only"],x["truth"]))
print("ERROR_ATTRIBUTION_START")
print(json.dumps({"total":len(keys),"paired":dict(paired),"top_families":summary[:30],
                  "negative_families":[x for x in summary if x["net_chardet_advantage"]<0][:15],
                  "mistake_patterns":Counter((r["truth"],str(r["cc_guess"])) for r in losses).most_common(25)},indent=2))
print("ERROR_ATTRIBUTION_END")
(root/"error_attribution.json").write_text(json.dumps({"paired":dict(paired),"families":summary,"chardet_only":losses,"codeccat_only":wins},indent=2))
