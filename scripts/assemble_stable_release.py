"""Assemble a Stable release from three successful tag-scoped CI runs.

No training, no source-corpus repackaging, no wheel tag rewriting. The tagged
source pins the version, supported platforms, and immutable numeric model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--linux", type=Path, required=True)
    p.add_argument("--platform", type=Path, required=True)
    p.add_argument("--sdist", type=Path, required=True)
    p.add_argument("--tagged-source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--linux-run", required=True)
    p.add_argument("--platform-run", required=True)
    p.add_argument("--sdist-run", required=True)
    args = p.parse_args()

    config = json.loads((args.tagged_source / "release-config.json").read_text())
    assert config["schema"] == "codeccat-stable-release-v1"
    version = config["version"]
    assert re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", version)
    assert tomllib.loads((args.tagged_source / "pyproject.toml").read_text())["project"]["version"] == version
    assert tomllib.loads((args.tagged_source / "native-prototype/Cargo.toml").read_text())["package"]["version"] == version
    model_sha = config["model_sha256"]
    assert re.fullmatch(r"[a-f0-9]{64}", model_sha)
    expected = {
        "manylinux_2_28_x86_64", "manylinux_2_28_aarch64",
        "macosx_10_12_x86_64", "macosx_11_0_arm64",
        "win_amd64", "win_arm64",
    }
    assert set(config["supported_platforms"]) == expected
    tag_commit = subprocess.check_output(
        ["git", "-C", str(args.tagged_source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert re.fullmatch(r"[a-f0-9]{40}", tag_commit)

    sources = {
        "linux": sorted(args.linux.rglob("*.whl")),
        "platform": sorted(args.platform.rglob("*.whl")),
        "sdist": sorted(args.sdist.rglob("*.tar.gz")),
    }
    assert len(sources["linux"]) == 2, sources["linux"]
    assert len(sources["platform"]) == 6, sources["platform"]
    assert len(sources["sdist"]) == 1, sources["sdist"]
    source_archive = sources["sdist"][0]
    assert source_archive.name == f"codeccat-{version}.tar.gz"
    with tarfile.open(source_archive, "r:gz") as archive:
        member = archive.extractfile(
            f"codeccat-{version}/native-prototype/python/codeccat/models/default.ccm"
        )
        assert member is not None and sha256(member.read()) == model_sha
        assert archive.getmember(f"codeccat-{version}/native-prototype/src/lib.rs").isfile()
        assert archive.getmember(f"codeccat-{version}/pyproject.toml").isfile()

    assembled = {}
    for wheel in sources["linux"]:
        assert "-manylinux_2_28_" in wheel.name
        assembled[wheel.name] = wheel
    for wheel in sources["platform"]:
        if "-linux_" in wheel.name or "-manylinux_2_28_" in wheel.name:
            # The historical dry-run supplies genuine manylinux Wheels in both
            # inputs; production six-platform CI supplies generic linux names.
            continue
        assert "-macosx_" in wheel.name or "-win_" in wheel.name, wheel
        assert wheel.name not in assembled, wheel.name
        assembled[wheel.name] = wheel
    assert len(assembled) == 6, sorted(assembled)

    tags = []
    audit = []
    for filename, source in sorted(assembled.items()):
        match = re.fullmatch(r"codeccat-" + re.escape(version) + r"-cp310-abi3-(.+)\.whl", filename)
        assert match, filename
        tags.append(match.group(1))
        with zipfile.ZipFile(source) as archive:
            assert archive.testzip() is None, filename
            names = archive.namelist()
            for entry in (
                "codeccat/models/default.ccm",
                "codeccat/models/default.ccm.meta.json",
                "codeccat/models/language-l3.json",
            ):
                assert entry in names, (filename, entry)
            digest = sha256(archive.read("codeccat/models/default.ccm"))
            assert digest == model_sha, (filename, digest)
            metadata = archive.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
            assert re.search(r"(?m)^Name: codeccat$", metadata)
            assert re.search(rf"(?m)^Version: {re.escape(version)}$", metadata)
            assert "Development Status :: 5 - Production/Stable" in metadata
            license_files = [n for n in names if n.endswith("/LICENSE")]
            assert license_files, filename
            assert "MIT License" in archive.read(license_files[0]).decode()
        audit.append({"file": filename, "bytes": source.stat().st_size,
                      "sha256": sha256(source.read_bytes()), "model_sha256": digest})
    assert set(tags) == expected and len(tags) == len(expected), (tags, expected)

    args.out.mkdir(parents=True, exist_ok=True)
    assert not list(args.out.iterdir()), "output directory must be empty"
    for source in assembled.values():
        shutil.copy2(source, args.out / source.name)
    shutil.copy2(source_archive, args.out / source_archive.name)
    provenance = {
        "version": version,
        "tag": "v" + version,
        "tag_commit": tag_commit,
        "model_sha256": model_sha,
        "training_source": config["training_source"],
        "training_commit": config["training_commit"],
        "manylinux_ci_run": args.linux_run,
        "cross_platform_ci_run": args.platform_run,
        "sdist_ci_run": args.sdist_run,
        "supported_wheel_tags": sorted(tags),
        "build_integrity": "Six wheels and sdist have separate CI and immutable model verification.",
        "rights_caveat": "See RIGHTS_AND_PROVENANCE.md for sources, Japanese training basis, and caveats.",
    }
    (args.out / "MODEL_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    files = sorted([*args.out.glob("*.whl"), *args.out.glob("*.tar.gz"),
                    args.out / "MODEL_PROVENANCE.json"])
    (args.out / "SHA256SUMS").write_text(
        "".join(f"{sha256(path.read_bytes())}  {path.name}\n" for path in files)
    )
    print("STABLE_RELEASE_ASSEMBLY_PASS", json.dumps({
        "version": version, "tag_commit": tag_commit, "model_sha256": model_sha,
        "wheels": audit, "sdist": source_archive.name, "asset_count": len(list(args.out.iterdir())),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
