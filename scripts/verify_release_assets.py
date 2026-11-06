"""Validate immutable Stable GitHub Release assets before any PyPI upload.

Read-only. A future release must carry release-config.json in its tagged
source and externally produced SHA256SUMS plus MODEL_PROVENANCE.json.
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


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(source: Path, assets: Path, tag: str, dest: Path | None = None) -> dict:
    if not re.fullmatch(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", tag):
        raise ValueError("Stable tag must be vMAJOR.MINOR.PATCH")
    config = json.loads((source / "release-config.json").read_text())
    assert config["schema"] == "codeccat-stable-release-v1"
    version = tag[1:]
    assert config["version"] == version
    assert tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"] == version
    assert tomllib.loads((source / "native-prototype/Cargo.toml").read_text())["package"]["version"] == version
    model_sha = config["model_sha256"]
    assert re.fullmatch(r"[a-f0-9]{64}", model_sha)
    expected = set(config["supported_platforms"])
    assert expected == {
        "manylinux_2_28_x86_64", "manylinux_2_28_aarch64",
        "macosx_10_12_x86_64", "macosx_11_0_arm64",
        "win_amd64", "win_arm64",
    }
    tag_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert re.fullmatch(r"[a-f0-9]{40}", tag_commit)
    provenance = json.loads((assets / "MODEL_PROVENANCE.json").read_text())
    assert provenance["tag"] == tag
    assert provenance["version"] == version
    assert provenance["tag_commit"] == tag_commit
    assert provenance["model_sha256"] == model_sha
    assert set(provenance["supported_wheel_tags"]) == expected

    lines = (assets / "SHA256SUMS").read_text().splitlines()
    recorded = {}
    for line in lines:
        match = re.fullmatch(r"([a-f0-9]{64})  ([A-Za-z0-9_.-]+)", line)
        if not match:
            raise ValueError("unsafe or malformed SHA256SUMS line")
        digest, name = match.groups()
        if name in recorded:
            raise ValueError("duplicate entry in SHA256SUMS")
        recorded[name] = digest
    wheel_paths = sorted(assets.glob("*.whl"))
    regex = re.compile(r"codeccat-" + re.escape(version) + r"-cp310-abi3-(.+)\.whl")
    seen = set()
    for wheel in wheel_paths:
        match = regex.fullmatch(wheel.name)
        assert match, wheel.name
        platform = match.group(1)
        assert platform in expected and platform not in seen
        seen.add(platform)
        with zipfile.ZipFile(wheel) as archive:
            assert archive.testzip() is None
            members = set(archive.namelist())
            assert "codeccat/models/default.ccm" in members
            assert "codeccat/models/default.ccm.meta.json" in members
            assert "codeccat/models/language-l3.json" in members
            assert hashlib.sha256(archive.read("codeccat/models/default.ccm")).hexdigest() == model_sha
            metadata = archive.read(next(n for n in members if n.endswith(".dist-info/METADATA"))).decode()
            assert re.search(r"(?m)^Name: codeccat$", metadata)
            assert re.search(rf"(?m)^Version: {re.escape(version)}$", metadata)
            assert any(n.endswith("/LICENSE") for n in members)
    assert seen == expected

    sdist = assets / ("codeccat-" + version + ".tar.gz")
    if sdist.is_file():
        with tarfile.open(sdist, "r:gz") as archive:
            member = archive.extractfile(
                f"codeccat-{version}/native-prototype/python/codeccat/models/default.ccm"
            )
            assert member is not None and hashlib.sha256(member.read()).hexdigest() == model_sha
    packages = {p.name for p in wheel_paths}
    if sdist.is_file():
        packages.add(sdist.name)
    files = {p.name for p in assets.iterdir() if p.is_file()}
    assert files == packages | {"MODEL_PROVENANCE.json", "SHA256SUMS"}, files
    assert set(recorded) == packages | {"MODEL_PROVENANCE.json"}
    for name, digest in recorded.items():
        assert checksum(assets / name) == digest, name
    if dest is not None:
        dest.mkdir(parents=True, exist_ok=True)
        for name in sorted(packages):
            shutil.copy2(assets / name, dest / name)
    return {"tag": tag, "commit": tag_commit, "model_sha256": model_sha,
            "wheels": len(wheel_paths), "sdist": sdist.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagged-source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print("VERIFIED_RELEASE_ASSETS", json.dumps(
        verify(args.tagged_source, args.assets, args.tag, args.out), sort_keys=True
    ))


if __name__ == "__main__":
    main()
