from __future__ import annotations
import json
from pathlib import Path

import pytest

from codeccat.model_io import load_bundle
from codeccat.api import Detector
from codeccat.training import fit


def prepare(root: Path):
    lines=[]
    examples=[
        ("utf-8", "日本語を含む UTF8 の文章です。" * 20,"utf-8"),
        ("utf-8-sig", "日本語で別の UTF8 BOM 文章です。" * 20,"utf-8-sig"),
        ("shift-jis", "日本語の符号化について考えています。"*20,"shift_jis"),
        ("euc-jp", "日本語の学術文章を掲載しています。"*20,"euc_jp"),
        ("utf-16-le", "Hello 世界 こんにちは" * 30,"utf-16-le"),
        ("utf-16-be", "Good morning 世界 こんにちは" * 30,"utf-16-be"),
    ]
    for split in ("train","validation"):
        for enc,s,codec in examples:
            for ix in range(3 if split=="train" else 1):
                # Different bytes across split, not only filenames.
                txt=s+f" {split} id {ix}."
                data=txt.encode(codec)
                filename=f"{split}_{enc}_{ix}.bin"
                (root/filename).write_bytes(data)
                lines.append({"path":filename,"encoding":enc,"split":split,
                              "group":f"{split}-group-{enc}"})
    manifest=root/"manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in lines)+"\n")
    return manifest


def test_training_roundtrip_and_group_barrier(tmp_path):
    manifest=prepare(tmp_path)
    out=tmp_path/"candidate.npz"
    report=tmp_path/"report.json"
    result=fit(manifest,out,report)
    assert out.exists() and report.exists()
    assert result["validation_n"]==6
    detector=Detector(load_bundle(out))
    assert detector.rank("test 語言".encode())
    assert result["feature_schema"]=="hmt768-baseline-518"
    assert result["warning"]


def test_reject_group_leakage(tmp_path):
    manifest=prepare(tmp_path)
    lines=[json.loads(z) for z in manifest.read_text().splitlines()]
    lines[-1]["group"]=lines[0]["group"]
    manifest.write_text("\n".join(json.dumps(r) for r in lines))
    with pytest.raises(ValueError,match="Group leakage"):
        fit(manifest,tmp_path/"x.npz",tmp_path/"r.json")


def test_reject_path_traversal(tmp_path):
    manifest=prepare(tmp_path)
    lines=[json.loads(z) for z in manifest.read_text().splitlines()]
    lines[0]["path"]="../escape.bin"
    manifest.write_text("\n".join(json.dumps(r) for r in lines))
    with pytest.raises(ValueError,match="unsafe relative"):
        fit(manifest,tmp_path/"x.npz",tmp_path/"r.json")
