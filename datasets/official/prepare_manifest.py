"""Produce metadata-only Stable23 input manifest from a *pinned local checkout*.

No third-party text is copied or uploaded. The manifest references files
inside the corpus checkout; it must remain alongside those input folders.
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE_COMMIT = "b0c0d206913ba118ffea2f7d306883e7a4a76db7"
ISO_SUFFIXES = set(
    "ar be bg br cs cy da de el en eo es et fa fi fr ga gd he hr hu "
    "id is it ja kk ko lt lv mk ms mt nl no pl pt ro ru sk sl sr sv "
    "tg th tr uk ur vi zh".split()
)
LANGUAGE_SUFFIXES = set(
    "arabic belarusian breton bulgarian chinese croatian czech danish "
    "dutch english esperanto estonian farsi finnish french gaelic german "
    "greek hebrew hungarian icelandic indonesian irish italian japanese "
    "kazakh korean latvian lithuanian macedonian malay maltese norwegian "
    "polish portuguese romanian russian serbian slovak slovene spanish "
    "swedish tajik thai turkish ukrainian urdu vietnamese welsh".split()
)


def encoding_from_dir(dirname: str) -> str | None:
    """Match the pinned source's parse_dir_name behavior without importing code."""
    if dirname not in ("None", "None-None", "ascii"):
        base, sep, suffix = dirname.rpartition("-")
        if sep and (
            suffix in LANGUAGE_SUFFIXES
            or (suffix in ISO_SUFFIXES and _canonical(base) is not None)
        ):
            dirname = base
    return _canonical(dirname)


def _canonical(name: str) -> str | None:
    try:
        return codecs.lookup(name).name
    except LookupError:
        return None


def source_group(path: Path) -> str:
    return (
        path.stem.lower()
        .replace("culturax_mc4_", "culturax_")
        .replace("culturax_oscar-2301_", "culturax_")
    )


def generate_records(root: Path, tracked: set[str] | None = None):
    """Preserve the historical sorted, strict-decode, prefix-dedup order."""
    root = root.resolve()
    seen = set()
    records = []
    for path in sorted(root.glob("*/*")):
        if not path.is_file():
            continue
        label = encoding_from_dir(path.parent.name)
        if label is None:
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"corpus path escapes checkout: {path}")
        relative = path.relative_to(root).as_posix()
        if tracked is not None and relative not in tracked:
            raise ValueError(f"untracked corpus input: {relative}")
        raw = path.read_bytes()
        if not raw:
            continue
        try:
            raw.decode(label, "strict")
        except UnicodeError:
            continue
        prefix_digest = hashlib.sha256(raw[:4096]).hexdigest()
        if prefix_digest in seen:
            continue
        seen.add(prefix_digest)
        group = source_group(path)
        fold = int(hashlib.sha256(group.encode()).hexdigest()[:12], 16) % 3
        records.append({
            "path": relative,
            "encoding": label,
            "split": "train",
            "group": group,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "prefix_sha256": prefix_digest,
            "historical_fold": fold,
        })
    return records


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, capture_output=True, text=True
    ).stdout.strip()


def prepare(corpus: Path, output: Path, *, expected_commit: str = SOURCE_COMMIT,
            require_official_size: bool = True) -> dict:
    root = corpus.resolve()
    actual = _git(root, "rev-parse", "HEAD").lower()
    if actual != expected_commit.lower():
        raise ValueError(f"wrong source commit: expected {expected_commit}, got {actual}")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked corpus files are modified")
    if output.resolve().parent != root:
        raise ValueError("manifest must be in the corpus checkout root")
    tracked = set(
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True, capture_output=True
        ).stdout.decode("utf-8").rstrip("\x00").split("\x00")
    )
    records = generate_records(root, tracked)
    classes = {row["encoding"] for row in records}
    if require_official_size and (len(records) < 1000 or len(classes) < 40):
        raise ValueError("insufficient verified historical training corpus")
    payload = "".join(
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for r in records
    )
    output.write_text(payload, encoding="utf-8")
    return {
        "source_commit": actual,
        "record_count": len(records),
        "class_count": len(classes),
        "manifest_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "output": str(output),
        "contains_raw_text": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a metadata-only pinned Stable23 input manifest")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-name", default="codeccat-replay-manifest.jsonl",
                        help="filename written at the corpus root")
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a filename, not a path")
    result = prepare(args.corpus, args.corpus / args.output_name)
    print("STABLE23_SOURCE_MANIFEST", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
