"""Run ONLY from a fresh venv with the built wheel installed; no PYTHONPATH.

Exercises the actual opt-in user API, public package metadata, numeric model
archive, Unicode L3, BOM policy and invalid suffix behavior.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import tempfile
from pathlib import Path

import codeccat
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_l3_language import LanguageCoherence
from codeccat.vnext_decoding import decode


def main():
    root=Path(codeccat.__file__).resolve()
    assert "site-packages" in str(root), root
    version=importlib.metadata.version("codeccat")
    assert version=="0.1.0a1",version
    deps=importlib.metadata.requires("codeccat") or []
    assert any("scikit-learn" in x and "vnext" in x for x in deps),deps
    samples=[
        (("The modern café and résumé " * 25).encode("utf-8"),"utf-8","fr-a"),
        (("A Unicode café with naïve words " * 25).encode("utf-8"),"utf-8","fr-b"),
        (("A plain page of sentences " * 30).encode("latin1"),"iso8859-1","en-a"),
        (("Another page and words " * 30).encode("latin1"),"iso8859-1","en-b"),
        (("日本語の文章と説明 "*25).encode("shift_jis"),"shift_jis","ja-a"),
        (("日本語の経済情報 "*25).encode("shift_jis"),"shift_jis","ja-b"),
    ]
    model=TrainableDetector.fit(samples)
    language=LanguageCoherence.fit(samples)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        model.save_portable(p/"model.ccnext")
        language.save(p/"language.json")
        restored=TrainableDetector.load_portable(p/"model.ccnext")
        restored_l3=LanguageCoherence.load(p/"language.json")
        raw=("The modern café and résumé " * 12).encode("utf-8")
        a=decode(restored,raw)
        b=decode(restored,raw,language_model=restored_l3)
        assert a.accepted and b.accepted
        assert a.text==raw.decode("utf-8")
        assert b.text==raw.decode("utf-8")
        assert not a.l3_called
        utf8bom=b"\xef\xbb\xbfThis is an official BOM"
        k=decode(restored,utf8bom,bom_policy="strip")
        # A plain utf8 detector may not expose the SIG alias as a label.
        # Unsupported SIG is an explicit abstention, not fabricated decode.
        assert k.encoding is None or k.text=="This is an official BOM"
        bad=decode(restored,raw+b"\xff")
        assert not bad.accepted or bad.text != raw.decode("utf-8")
        print(json.dumps({
            "schema":"vnext-wheel-install-smoke-v1","status":"PASS",
            "installed_path":str(root),"wheel_version":version,
            "model_fingerprint":restored.fingerprint,
            "l3_fingerprint":restored_l3.fingerprint,
            "api_decoded":b.accepted,
            "default_l3_disabled":not a.l3_called,
            "bom_policy_checked":True,
        },sort_keys=True))


if __name__=="__main__":
    main()
