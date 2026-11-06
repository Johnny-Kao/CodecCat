"""Assemble a verifiable Stable GitHub release from tag-scoped CI Wheels.

Never renames a Linux wheel to claim manylinux: those bytes must originate
from the genuine manylinux build workflow.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

VERSION = "0.1.0"
FROZEN_MODEL = "16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b"
EXPECTED = (
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "macosx_10_12_x86_64",
    "macosx_11_0_arm64",
    "win_amd64",
    "win_arm64",
)

def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--linux",type=Path,required=True)
    p.add_argument("--platform",type=Path,required=True)
    p.add_argument("--tagged-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--linux-run",required=True)
    p.add_argument("--platform-run",required=True)
    args=p.parse_args()
    assert "version = \"0.1.0\"" in (args.tagged_source/"pyproject.toml").read_text()
    tag_commit=subprocess.check_output(
        ["git","-C",str(args.tagged_source),"rev-parse","HEAD"],text=True
    ).strip()
    assert len(tag_commit)==40
    sources = {
        "linux": sorted(args.linux.rglob("*.whl")),
        "platform": sorted(args.platform.rglob("*.whl")),
    }
    assert len(sources["linux"])==2,sources["linux"]
    assert len(sources["platform"])==6,sources["platform"]
    assembled={}
    for wheel in sources["linux"]:
        assert "-manylinux_2_28_" in wheel.name
        assembled[wheel.name]=wheel
    for wheel in sources["platform"]:
        if "-linux_" in wheel.name:
            continue
        assert "-macosx_" in wheel.name or "-win_" in wheel.name,wheel
        assert wheel.name not in assembled,wheel
        assembled[wheel.name]=wheel
    assert len(assembled)==6,sorted(assembled)
    tags=[]
    audit=[]
    for filename,source in sorted(assembled.items()):
        match=re.fullmatch(r"codeccat-0\.1\.0-cp310-abi3-(.+)\.whl",filename)
        assert match,filename
        tags.append(match.group(1))
        with zipfile.ZipFile(source) as z:
            assert z.testzip() is None,filename
            names=z.namelist()
            assert "codeccat/models/default.ccm" in names,filename
            assert "codeccat/models/default.ccm.meta.json" in names,filename
            assert "codeccat/models/language-l3.json" in names,filename
            digest=sha256(z.read("codeccat/models/default.ccm"))
            assert digest==FROZEN_MODEL,(filename,digest)
            metadata=z.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
            assert "Name: codeccat" in metadata
            assert "Version: 0.1.0" in metadata
            assert "Development Status :: 5 - Production/Stable" in metadata
            license_files=[n for n in names if n.endswith("/LICENSE")]
            assert license_files,filename
            assert "MIT License" in z.read(license_files[0]).decode()
        audit.append({"file":filename,"bytes":source.stat().st_size,
                      "sha256":sha256(source.read_bytes()),"model_sha256":digest})
    assert set(tags)==set(EXPECTED),(tags,EXPECTED)
    args.out.mkdir(parents=True,exist_ok=True)
    for source in assembled.values():
        shutil.copy2(source,args.out/source.name)
    provenance={
        "version":VERSION,"tag":"v0.1.0","tag_commit":tag_commit,
        "model_sha256":FROZEN_MODEL,
        "model_reference_archive":"v0.1.0a4/native-reference-v0.1.0a4.tar.gz",
        "training_source":"chardet/test-data",
        "training_commit":"b0c0d206913ba118ffea2f7d306883e7a4a76db7",
        "manylinux_ci_run":args.linux_run,
        "cross_platform_ci_run":args.platform_run,
        "supported_wheel_tags":list(tags),
        "build_integrity":"Every Wheel was audited and passed separate installed-wheel CI.",
        "rights_caveat":"See RIGHTS_AND_PROVENANCE.md. Training-data rights not independently cleared.",
    }
    (args.out/"MODEL_PROVENANCE.json").write_text(
        json.dumps(provenance,indent=2,sort_keys=True)+"\n"
    )
    entries=sorted([*args.out.glob("*.whl"),args.out/"MODEL_PROVENANCE.json"])
    (args.out/"SHA256SUMS").write_text(
        "".join(f"{sha256(f.read_bytes())}  {f.name}\n" for f in entries)
    )
    print("STABLE_RELEASE_ASSEMBLY_PASS",json.dumps({
        "tag_commit":tag_commit,"wheels":audit,"support":sorted(tags),
        "asset_count":len(list(args.out.iterdir())),
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
