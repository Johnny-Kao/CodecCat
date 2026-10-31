"""Freeze untouched independently-maintained legacy translation file bytes.

Sources and selection are fixed here WITHOUT importing any detector or fitting
a model. These are source-controlled raw PO byte windows, not synthetic encodes.
The sources' copyright terms remain with the original maintainers; no payload
bytes are committed to CodecCat, and all are TEST ONLY.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_vim_catalog_freeze import catalog_samples,sha

SOURCES=(
    ("lynx", "7f670781d769903d5a32a4b9eb68175f2e687a26",("po",)),
    ("gettext","1bc095c8f78b4ff07b48535033e504c126624f3c",("gettext-runtime/po","gettext-tools/po")),
)
MAX_UTF8_CATALOGS_PER_SOURCE=6
SEED="codeccat-independent-legacy-po-20261101"
LEGACY_MIN_WINDOWS=4


def git_sha(folder):
    return subprocess.check_output(["git","-C",str(folder),"rev-parse","HEAD"],
                                   text=True).strip()


def select_catalogs(label,root,locations):
    found=[]
    for location in locations:
        for file in (root/location).glob("*.po"):
            samples, ledger=catalog_samples(file)
            ledger["path"]=file.relative_to(root).as_posix()
            for sample in samples:
                sample["path"]=ledger["path"]
            found.append((ledger,samples))
    legacy=[]
    utf=[]
    rejected=[]
    for ledger,samples in found:
        if not samples:
            rejected.append(ledger)
            continue
        if samples[0]["codec"]=="utf-8":
            utf.append((ledger,samples))
        else:
            legacy.append((ledger,samples))
    # Control files from precommitted stable selection, independent of model scores.
    utf.sort(key=lambda x:sha((SEED+label+x[0]["path"]).encode()))
    utf=utf[:MAX_UTF8_CATALOGS_PER_SOURCE]
    return legacy+utf,rejected,len(found),len(legacy),len(utf)


def freeze(args):
    root_map={"lynx":Path(args.lynx).resolve(),"gettext":Path(args.gettext).resolve()}
    records=[]
    metadata={}
    for name,sha_expected,locations in SOURCES:
        root=root_map[name]
        if git_sha(root)!=sha_expected:
            raise RuntimeError(f"{name}: source revision differs from pinned SHA")
        selected,rejected,scanned,nlegacy,nutf=select_catalogs(name,root,locations)
        metadata[name]={
            "revision":sha_expected,
            "paths":list(locations),
            "scanned_catalogs":scanned,
            "selected_legacy_catalogs":nlegacy,
            "selected_utf8_control_catalogs":nutf,
            "rejected_catalogs":len(rejected),
        }
        for ledger,samples in selected:
            records.append({"source":name,"catalog":ledger,"windows":samples})
    legacy_count=sum(len(row["windows"]) for row in records
                     if row["catalog"].get("codec")!="utf-8")
    if legacy_count<LEGACY_MIN_WINDOWS:
        raise RuntimeError(f"too little legacy coverage: {legacy_count}")
    fingerprint=sha(json.dumps(records,sort_keys=True).encode())
    report={
        "schema":"codeccat-legacy-po-external-frozen-v1",
        "pre_registered":True,
        "source_revisions":metadata,
        "selection_seed":SEED,
        "selection_rule":"ALL strictly decoded non-UTF8 source-controlled .po + 6 stable-hash UTF8 control catalogs per repository",
        "role":"TEST_ONLY_NEVER_TRAIN",
        "fingerprint":fingerprint,
        "recorded":records,
        "n_catalogs":len(records),
        "n_legacy_windows":legacy_count,
        "n_utf8_windows":sum(len(r["windows"]) for r in records if r["catalog"].get("codec")=="utf-8"),
        "by_codec":dict(sorted(Counter(w["codec"] for row in records for w in row["windows"]).items())),
        "limitations":[
            "Some source catalogs may be converted/generated and are not proof of originally authored encoding",
            "Multiple windows per document are correlated; do not claim independent observations",
            "Only two external code bases; encoding coverage may be small",
            "No inferred license to redistribute native corpus bytes or train on these sources",
        ],
    }
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"source_manifest.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("FROZEN_LEGACY_SOURCE_BEGIN",flush=True)
    print(json.dumps({k:report[k] for k in ("fingerprint","source_revisions","n_catalogs","n_legacy_windows","n_utf8_windows","by_codec")},indent=2,sort_keys=True),flush=True)
    print("FROZEN_LEGACY_SOURCE_END",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--lynx",required=True)
    p.add_argument("--gettext",required=True)
    p.add_argument("--out",default="research-results/legacy-po-freeze")
    freeze(p.parse_args())
