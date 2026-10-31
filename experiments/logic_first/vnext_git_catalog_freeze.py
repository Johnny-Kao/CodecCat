"""Prerequisite: freeze fresh Git translation catalogs WITHOUT model scoring.

The code and min-support threshold are inherited unchanged from earlier Vim
experiment; this independent repository is only a separate confirmation set.
Git's PO bytes are never redistributed or added to model training.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,pathlib,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from vnext_vim_catalog_freeze import catalog_samples,sha,WINDOW,POSITIONS,MIN_NONASCII,SNIPPET_SCHEME

GIT_REV="50fe5cc63f9cd6e89b87a2c9ccf96a444f4cfdb7"

def run(args):
    root=pathlib.Path(args.source)
    rev=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if rev!=GIT_REV:raise RuntimeError("pinned Git translation source revision changed")
    catalogs=[];samples=[]
    for file in sorted((root/"po").glob("*.po")):
        subset, rec=catalog_samples(file)
        catalogs.append(rec)
        samples.extend(subset)
    if len(samples)<60 or sum(int(c.get("windows",0)>0) for c in catalogs)<17:
        raise RuntimeError("insufficient independent eligible PO sample files")
    manifest={
        "schema":"codeccat-git-real-po-source-freeze-v1",
        "source":"git/git",
        "revision":GIT_REV,
        "role":"NEW_EXTERNAL_BLIND_SOURCE; NEVER_TRAIN",
        "source_license":"Git project license; test only, do not redistribute original bytes",
        "extractor":SNIPPET_SCHEME,
        "window":WINDOW,"positions":POSITIONS,"min_nonascii":MIN_NONASCII,
        "n_catalogs":len(catalogs),"n_windows":len(samples),
        "catalogs":catalogs,"windows":samples,
        "disclosure":"The source Git translations are naturally versioned files but each catalog's windows are correlated.",
    }
    manifest["sha256"]=sha(json.dumps(manifest,sort_keys=True).encode())
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"source_manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print("GIT_PO_FREEZE_RESULT_BEGIN")
    print(json.dumps({"sha256":manifest["sha256"],"revision":rev,
                      "catalogs":len(catalogs),"windows":len(samples),
                      "valid_catalogs":sum(bool(c.get("windows")) for c in catalogs),
                      "rejected":{c["path"]:c.get("reason") for c in catalogs if not c.get("windows")}},
                     sort_keys=True,indent=2))
    print("GIT_PO_FREEZE_RESULT_END")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--out",default="research-results/git-po-freeze")
    run(p.parse_args())
