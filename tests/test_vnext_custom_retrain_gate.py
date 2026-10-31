"""Contract checks for the research-only custom retraining intake/gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "logic_first"))
from vnext_custom_retrain_gate import load_manifest, run


def case(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    records = [
        ("t/ja1.bin", "utf-8", "source-ja-1", "train", "日本語の文章が含まれています。" * 12),
        ("t/ja2.bin", "utf-8", "source-ja-2", "train", "日本の社会や文化に関する記事。" * 12),
        ("t/ru1.bin", "cp1251", "source-ru-1", "train", "Это русский текст об истории." * 12),
        ("t/ru2.bin", "cp1251", "source-ru-2", "train", "Русский язык и новая информация." * 12),
        ("v/ja3.bin", "utf-8", "unseen-ja", "validation", "独立した日本語の評価文章です。" * 12),
        ("v/ru3.bin", "cp1251", "unseen-ru", "validation", "Статья на русском языке." * 12),
    ]
    rows = []
    for path, enc, group, split, text in records:
        f = root / path
        f.parent.mkdir(exist_ok=True, parents=True)
        f.write_bytes(text.encode(enc))
        rows.append({
            "path": path, "encoding": enc, "source_group": group,
            "split": split, "origin": "test-owned-data", "rights": "owned",
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root, manifest, rows


def write_manifest(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_clean_custom_corpus_builds_roundtrip_candidate_without_promoting(tmp_path):
    root, manifest, rows = case(tmp_path)
    dataset, ledger = load_manifest(root, manifest)
    assert ledger["n_train"] == 4
    assert ledger["n_validation"] == 2
    assert len(dataset) == 6
    out = tmp_path / "result"
    run(SimpleNamespace(root=str(root), manifest=str(manifest), out=str(out),
                        policy="residual", incumbent=None))
    report = json.loads((out / "report.json").read_text())
    assert report["promotion_allowed"] is False
    assert report["decision"] == "RESEARCH_CANDIDATE_ONLY"
    assert report["validation"]["n"] == 2
    assert report["source_ledger"]["n_train"] == 4
    assert report["model"]["n_classes"] == 2
    assert (out / "candidate.ccnext").is_file()


def test_cross_encoding_same_text_cannot_cross_validation_boundary(tmp_path):
    root, manifest, rows = case(tmp_path)
    # Same source text, distinct raw byte sequences and different group ids.
    (root / "t" / "latin.bin").write_bytes("café from a document".encode("cp1252"))
    (root / "v" / "latin.bin").write_bytes("café from a document".encode("utf-8"))
    rows.extend([
        {"path": "t/latin.bin", "encoding": "cp1252", "source_group": "extra-train",
         "split": "train", "origin": "owned", "rights": "owned"},
        {"path": "v/latin.bin", "encoding": "utf-8", "source_group": "extra-valid",
         "split": "validation", "origin": "owned", "rights": "owned"},
    ])
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="same decoded document content"):
        load_manifest(root, manifest)


def test_group_leak_rejected(tmp_path):
    root, manifest, rows = case(tmp_path)
    rows[-1]["source_group"] = rows[0]["source_group"]
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="disjoint"):
        load_manifest(root, manifest)


def test_file_path_escape_rejected(tmp_path):
    root, manifest, rows = case(tmp_path)
    rows[0]["path"] = "../manifest.jsonl"
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="traverse"):
        load_manifest(root, manifest)


def test_rights_and_source_are_required(tmp_path):
    root, manifest, rows = case(tmp_path)
    del rows[0]["rights"]
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="missing path/encoding"):
        load_manifest(root, manifest)


def test_raw_byte_duplicate_rejected(tmp_path):
    root, manifest, rows = case(tmp_path)
    duplicate = root / "v" / "copy.bin"
    duplicate.write_bytes((root / "t" / "ja1.bin").read_bytes())
    rows.append({"path": "v/copy.bin", "encoding": "utf-8",
                 "source_group": "new-document-group", "split": "validation",
                 "origin": "owned", "rights": "owned"})
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="repeated identical raw bytes"):
        load_manifest(root, manifest)


def test_invalid_labeled_encoding_is_rejected(tmp_path):
    root, manifest, rows = case(tmp_path)
    (root / "t" / "ja1.bin").write_bytes(b"\xff\xfe\xff")
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="invalid labeled byte stream"):
        load_manifest(root, manifest)
