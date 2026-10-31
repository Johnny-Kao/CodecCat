"""Ground-truth audit of source-controlled PO fixture charset declarations.

NO MODELS: diagnose whether legacy charset labels contradict the original
translation bytes. Strict decode alone is NOT proof that a single-byte label
is true (Latin-1 can decode every byte).

Do not change candidate weights or retroactively discard evaluated cases.
Output is a provenance warning; source maintainers are authoritative.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import sys
import unicodedata

sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_new_gnu_source_freeze import SOURCES, SCHEMA
from vnext_vim_catalog_freeze import (
    catalog_samples,fragments,take_window,POSITIONS,WINDOW,sha,
)

EXPECTED_SOURCE_SHA="7e11e0d76121103b6e30ba23f3f64e0416f15b65214efc76fc7b50edf098300e"


def utf8_analysis(raw:bytes, claimed:str) -> dict:
    report={"valid_utf8":False,"utf8_nonascii":0,
            "utf8_controls":None,"declared_nonascii":0,
            "legacy_mojibake_pairs":0,"utf8_text_sha":None}
    try:
        legacy=raw.decode(claimed,"strict")
    except (LookupError,UnicodeError) as ex:
        raise ValueError("frozen reference became invalid") from ex
    report["declared_nonascii"]=sum(ord(c)>127 for c in legacy)
    report["legacy_mojibake_pairs"]=sum(
        a in ("\u00c2","\u00c3") and 0x80<=ord(b)<=0xbf
        for a,b in zip(legacy,legacy[1:])
    )
    try:
        text=raw.decode("utf-8","strict")
    except UnicodeDecodeError:
        return report
    report["valid_utf8"]=True
    report["utf8_text_sha"]=sha(text.encode("utf-8","surrogatepass"))
    report["utf8_nonascii"]=sum(ord(c)>127 for c in text)
    report["utf8_controls"]=sum(unicodedata.category(c)=="Cc" and
                                 c not in ("\r","\n","\t") for c in text)
    return report


def audit(args):
    roots={"gawk":Path(args.gawk).resolve(),"bash":Path(args.bash).resolve()}
    for name,(revision,_) in SOURCES.items():
        observed=subprocess.check_output(["git","-C",str(roots[name]),"rev-parse","HEAD"],text=True).strip()
        if observed!=revision:
            raise RuntimeError("source revision changed")
    manifest=json.loads(Path(args.manifest).read_text())
    if manifest.get("source_digest")!=EXPECTED_SOURCE_SHA or manifest.get("schema")!=SCHEMA:
        raise RuntimeError("unknown preregistered manifest")
    per_catalog=[]
    counts=Counter()
    for x in manifest["records"]:
        source=x["source"]
        record=x["catalog"]
        f=(roots[source]/record["path"]).resolve()
        if not f.is_relative_to(roots[source]):
            raise RuntimeError("untrusted source path")
        windows,meta=catalog_samples(f)
        meta["path"]=record["path"]
        for w in windows:w["path"]=record["path"]
        if meta!=record or windows!=x["windows"]:
            raise RuntimeError("frozen source changed")
        data=fragments(f.read_bytes())
        checks=[]
        for w in windows:
            start=round((len(data)-WINDOW)*POSITIONS[w["window"]])
            raw=take_window(data,max(0,start),w["codec"])
            if raw is None or sha(raw)!=w["sha256"]:
                raise RuntimeError("window bytes changed")
            chk=utf8_analysis(raw,w["codec"])
            chk["window"]=w["window"]
            chk["byte_sha"]=w["sha256"]
            checks.append(chk)
        n_valid=sum(z["valid_utf8"] for z in checks)
        n_utf8_nonascii=sum(z["utf8_nonascii"] for z in checks)
        n_mojibake=sum(z["legacy_mojibake_pairs"] for z in checks)
        # Suspicion, NOT a claim of a mislabeled file. This gate never
        # excludes or rewrites evidence automatically.
        flagged=(record["codec"]!="utf-8" and n_valid>=2 and
                 n_utf8_nonascii>=2 and n_mojibake>=1)
        counts["suspect_catalogs"]+=int(flagged)
        counts["legacy_catalogs"]+=int(record["codec"]!="utf-8")
        per_catalog.append({
            "source":source,"file":record["path"],
            "declared_codec":record["codec"],
            "file_sha":record["file_sha256"],
            "windows":len(checks),"windows_strict_utf8":n_valid,
            "utf8_nonascii_points":n_utf8_nonascii,
            "legacy_mojibake_pairs":n_mojibake,
            "potential_charset_mismatch":flagged,
            "evidence":checks,
        })
    report={
        "schema":"codeccat-native-po-charset-integrity-v1",
        "data_role":"LABEL_QUALITY_AUDIT_ONLY; NO_MODEL_SCORING",
        "source_selection_sha":EXPECTED_SOURCE_SHA,
        "counts":dict(counts),
        "catalogs":per_catalog,
        "interpretation":[
            "Strictly decoding a single-byte codec proves little about source encoding identity",
            "UTF8-valid original bytes with a different single-byte declaration can be intentional or stale metadata",
            "Moji-bake pair evidence increases suspicion but does not justify overriding source labels automatically",
            "No previously measured benchmark result has been retrospectively changed or excluded",
        ],
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"audit.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("NATIVE_LABEL_QUALITY_AUDIT_BEGIN",flush=True)
    print(json.dumps({"counts":dict(counts),
                      "potential_mismatches":[
                        {k:z[k] for k in ("source","file","declared_codec","windows","windows_strict_utf8",
                                         "utf8_nonascii_points","legacy_mojibake_pairs")}
                        for z in per_catalog if z["potential_charset_mismatch"]],
                      "legacy_controls":[
                        {k:z[k] for k in ("source","file","declared_codec","windows_strict_utf8",
                                         "utf8_nonascii_points","legacy_mojibake_pairs")}
                        for z in per_catalog if z["declared_codec"]!="utf-8"]},
                     indent=2,sort_keys=True),flush=True)
    print("NATIVE_LABEL_QUALITY_AUDIT_END",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gawk",required=True)
    p.add_argument("--bash",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",default="research-results/native-po-label-audit")
    audit(p.parse_args())
