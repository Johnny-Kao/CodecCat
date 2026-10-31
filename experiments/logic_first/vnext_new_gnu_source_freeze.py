"""Pre-register a NEW Gawk/Bash catalog source, no model scores or training.

This source is independent of the already inspected Lynx/gettext/Git/Vim suites.
Selection policy is exactly the earlier code: all validated legacy files and
a fixed stable-hash six UTF-8 controls per upstream project.
"""
from __future__ import annotations
import argparse,collections,json,pathlib,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from vnext_legacy_po_freeze import select_catalogs,sha

SOURCES={
  "gawk":("455c99c010eafca5729081954d23059e3cbca571",("po",)),
  "bash":("9439ce094c9aa7557a9d53ac7b412a23aa66e36b",("po",)),
}
SCHEMA="codeccat-frozen-unseen-gnu-po-v1"


def run(args):
    root_map={"gawk":pathlib.Path(args.gawk).resolve(),
              "bash":pathlib.Path(args.bash).resolve()}
    included=[];summary={}
    for name,(revision,paths) in SOURCES.items():
        root=root_map[name]
        sha_actual=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        if sha_actual!=revision:raise RuntimeError("source revision mismatch")
        chosen,rejected,scanned,nlegacy,ncontrol=select_catalogs(name,root,paths)
        summary[name]={"revision":revision,"scanned":scanned,
                       "legacy_catalogs":nlegacy,"control_catalogs":ncontrol,
                       "rejected_catalogs":len(rejected)}
        for ledger,windows in chosen:
            included.append({"source":name,"catalog":ledger,"windows":windows})
    counts=collections.Counter(w["codec"] for item in included for w in item["windows"])
    total=sum(counts.values())
    legacy=total-counts.get("utf-8",0)
    if legacy<4 or counts.get("utf-8",0)<20:
        raise RuntimeError(f"not sufficient nonUTF8 vs UTF8 controls: {dict(counts)}")
    report={"schema":SCHEMA,"manifest_role":"NEW_TEST_ONLY_NEVER_TRAIN",
            "selection_protocol":"all non-utf8 validated catalogs plus 6 stable-hash utf8 controls per source",
            "sources":summary,
            "n_catalogs":len(included),"n_windows":total,
            "legacy_windows":legacy,"utf8_windows":counts.get("utf-8",0),
            "by_codec":dict(sorted(counts.items())),
            "records":included,
            "caveats":[
                "Native source-controlled translation bytes; some may be generated from other files",
                "Four sampled windows per file are correlated",
                "No original bytes distributed or added to any training model",
                "This is a diagnostic of across-source generalization, not population accuracy",
            ]}
    report["source_digest"]=sha(json.dumps(included,sort_keys=True).encode())
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"manifest.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("NEW_NATIVE_SOURCE_FROZEN_BEGIN",flush=True)
    print(json.dumps({k:report[k] for k in ("sources","source_digest","n_catalogs","legacy_windows","utf8_windows","by_codec")},sort_keys=True,indent=2),flush=True)
    print("NEW_NATIVE_SOURCE_FROZEN_END",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gawk",required=True)
    p.add_argument("--bash",required=True)
    p.add_argument("--out",default="research-results/new-gnu-source-freeze")
    run(p.parse_args())
