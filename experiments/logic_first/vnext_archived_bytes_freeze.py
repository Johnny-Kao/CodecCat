"""Freeze independent bytes from the historical BYVoid/uchardet suite.

This workflow intentionally does NOT fit any model or score detectors.
The archive is test-only and the original bytes are not included in CodecCat.
"""
from __future__ import annotations
import argparse, codecs, collections, hashlib, json, pathlib, subprocess

EXPECTED_SHA = "4e685757780cb3c652fc6c9ec759f62888969ec9"
ALIASES = {
    "windows-1250":"cp1250","windows-1251":"cp1251","windows-1252":"cp1252",
    "windows-1253":"cp1253","windows-1255":"cp1255","windows-1256":"cp1256",
    "windows-1258":"cp1258","ibm855":"cp855","ibm866":"cp866",
    "mac-cyrillic":"mac-cyrillic","ascii":"ascii"
}

def label_for(name):
    name=ALIASES.get(name,name)
    try: return codecs.lookup(name).name
    except LookupError: return None

def run(args):
    root=pathlib.Path(args.fixtures)
    actual=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if actual!=EXPECTED_SHA: raise RuntimeError("upstream fixture revision changed")
    rows=[]
    for p in sorted((root/"test").glob("*/*.txt")):
        if not p.is_file(): continue
        enc=p.stem.lower()
        label=label_for(enc)
        raw=p.read_bytes()
        report={"path":p.relative_to(root).as_posix(),"lang":p.parent.name,
                "encoding_tag":enc,"python_codec":label,"sha256":hashlib.sha256(raw).hexdigest(),
                "bytes":len(raw)}
        if label is None:
            report["classification"]="unsupported_Python_codec"
        else:
            try:
                decoded=raw.decode(label,"strict")
                report["classification"]="valid_tagged_bytes"
                report["decoded_sha256"]=hashlib.sha256(decoded.encode("utf-8","surrogatepass")).hexdigest()
            except (UnicodeError,LookupError):
                report["classification"]="invalid_under_filename_codec"
        rows.append(report)
    if len(rows)<50: raise RuntimeError(f"incomplete corpus: {len(rows)} files")
    counts=collections.Counter(r["classification"] for r in rows)
    valid=[r for r in rows if r["classification"]=="valid_tagged_bytes"]
    if len(valid)<43: raise RuntimeError(f"insufficient valid tagged sources: {len(valid)}")
    fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    manifest={
        "schema":"codeccat-frozen-independent-uchardet-v1",
        "source":"BYVoid/uchardet",
        "origin_revision":actual,
        "data_role":"TEST_ONLY_NO_TRAINING",
        "data_license":"not separately verified; do not redistribute file contents or train weights",
        "provenance_caveat":"Original archived test bytes labeled by filenames; not proven naturally authored in that encoding",
        "fingerprint":fingerprint,
        "all_files":len(rows),"valid_files":len(valid),
        "by_classification":dict(counts),"languages":sorted({r["lang"] for r in rows}),
        "files":rows,
    }
    dest=pathlib.Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/"frozen_manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print("FROZEN_ARCHIVED_BYTES_BEGIN")
    print(json.dumps({"origin_revision":actual,"fingerprint":fingerprint,"all_files":len(rows),
                      "valid_files":len(valid),"by_classification":dict(counts),
                      "languages":manifest["languages"]},sort_keys=True,indent=2))
    print("FROZEN_ARCHIVED_BYTES_END")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fixtures",required=True)
    ap.add_argument("--out",default="research-results/independent-archived-bytes-freeze")
    run(ap.parse_args())
