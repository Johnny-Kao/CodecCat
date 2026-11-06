"""Pre-release native wheel reliability and compatibility gates.

Run in an isolated environment containing ONLY the built codeccat wheel and
Python stdlib. Do not import legacy sklearn, NumPy, or training code.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
from collections import Counter

from codeccat import CodecCat, decode, detect
from codeccat._codeccat_native import NativeL2
from codeccat.vnext_decoding import Detection

MODEL = pathlib.Path("native-prototype/model/stable23.ccm")
F32 = pathlib.Path("native-prototype/python/codeccat/models/default.ccm")


def expect_error(payload: bytes, label: str) -> None:
    try:
        NativeL2.from_model_bytes(payload)
    except (ValueError, OverflowError, MemoryError, OSError):
        return
    raise AssertionError(f"bad model silently accepted: {label}")


def repair_digest(buffer: bytearray) -> bytes:
    buffer[-32:] = hashlib.sha256(buffer[:-32]).digest()
    return bytes(buffer)


def model_security() -> None:
    original = F32.read_bytes()
    assert original[:4] == b"CCM2"
    assert NativeL2.from_model_file(str(F32)).numeric_format() == "f32"
    assert NativeL2.from_model_file(str(MODEL)).numeric_format() == "f64"

    expect_error(b"", "empty")
    expect_error(b"CCM2", "short-header")
    expect_error(original[:100], "truncated")
    expect_error(original + b"extra", "trailing-bytes")
    expect_error(original[:-1], "missing-digest")
    bad = bytearray(original)
    bad[12] ^= 255
    expect_error(bytes(bad), "corrupt-without-updated-digest")

    mutations = [
        ("wrong-magic", 0, 0),
        ("zero-features", 4, 0),
        ("excessive-classes", 8, 255),
        ("invalid-ngram-min", 12, 0),
        ("invalid-ngram-max", 13, 9),
        ("unsupported-flags", 14, 255),
        ("nonzero-reserved", 15, 255),
    ]
    for name, offset, value in mutations:
        changed = bytearray(original)
        changed[offset] = value
        expect_error(repair_digest(changed), name)

    # Fuzz the length/header parser, including valid checksums for altered
    # fields. All malformed inputs must raise rather than panic or allocate
    # proportional to untrusted dimensions.
    for i in range(80):
        changed = bytearray(original[:200 + i])
        if len(changed) < 48:
            continue
        changed[-32:] = hashlib.sha256(changed[:-32]).digest()
        expect_error(bytes(changed), f"bounded-truncation-{i}")
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "bad.ccm"
        p.write_bytes(b"invalid")
        try:
            NativeL2.from_model_file(str(p))
        except (ValueError, OSError):
            pass
        else:
            raise AssertionError("bad mmap model file accepted")
    print("RELEASE_MODEL_SECURITY_PASS", json.dumps({"malformed_cases": 7 + len(mutations) + 80}))


def api_contract() -> None:
    from codeccat import __all__
    expected = {"CodecCat", "detect", "decode"}
    assert expected <= set(__all__)
    for dep in ("numpy", "scipy", "sklearn", "joblib"):
        assert importlib.util.find_spec(dep) is None, dep
    dist = importlib.metadata.distribution("codeccat")
    requirements = dist.requires or []
    assert all("extra == " in r and "legacy" in r for r in requirements), requirements
    assert dist.version == "0.1.0a4", dist.version
    assert CodecCat.__module__ == "codeccat.usable"
    assert detect(b"hello").text == "hello"
    assert decode(b"hello") == "hello"

    for value in ("hello", None, 5, {"a": "b"}):
        try:
            detect(value)
        except TypeError:
            pass
        else:
            raise AssertionError(f"accepted unexpected input type {type(value)}")
    for raw in (bytearray(b"hello"), memoryview(b"hello"), b"hello"):
        assert detect(raw).text == "hello"

    default = CodecCat()
    custom = CodecCat.from_files(F32)
    assert default.detect(b"hello") == custom.detect(b"hello")
    assert isinstance(default.detect(b"hello"), Detection)
    for policy in ("strip", "preserve"):
        output = default.detect(b"\xef\xbb\xbfhello", bom_policy=policy)
        assert output.text is not None and output.encoding is not None

    # Each old top-level symbol remains discoverable and has a clear,
    # actionable missing-extra error rather than disappearing silently.
    legacy = (
        "DetectionResult", "Detector", "LinearModel", "RuntimeBundle",
        "RuntimeStack", "load_bundle", "save_bundle"
    )
    import codeccat
    for name in legacy:
        assert name in codeccat.__all__
        try:
            getattr(codeccat, name)
        except ImportError as exc:
            assert "codeccat[legacy]" in str(exc), (name, str(exc))
        else:
            raise AssertionError(f"legacy {name} unexpectedly loaded NumPy in clean env")
    print("RELEASE_LEGACY_OPTIONAL_IMPORT_PASS",
          json.dumps({"symbols":len(legacy),"requires_extra":"legacy"}))
    print("RELEASE_API_BASE_PASS")


def utf8_evidence_contract() -> None:
    from codeccat.vnext_decoding import _strong_utf8_evidence, decode as decode_raw
    from codeccat.vnext_strong_l2 import Ranked

    text = "こんにちは café résumé " * 15
    raw = text.encode("utf-8")
    assert len(raw) > 64 and _strong_utf8_evidence(raw)
    assert not _strong_utf8_evidence(b"hello ASCII" * 30)
    assert not _strong_utf8_evidence(raw + b"\xff")
    assert not _strong_utf8_evidence(b"\xef\xbb\xbf" + raw)
    assert not _strong_utf8_evidence(("hello " * 200 + "\x01" + text).encode("utf-8"))
    assert not _strong_utf8_evidence(("café " * 2).encode("utf-8"))

    class LatinRanking:
        class classifier:
            classes_ = ("iso8859-1", "utf-8")
        def rank(self, value, *, final=True, structural_first=False):
            return Ranked("iso8859-1", ("iso8859-1", "utf-8"),
                          (1.0, 0.5), None, False)

    result = decode_raw(LatinRanking(), raw)
    assert result.text == text
    assert result.encoding == "utf-8"
    assert result.reason == "utf8_evidence"
    assert result.ambiguous and result.candidates[0] == "utf-8"

    class NoUnicode(LatinRanking):
        class classifier:
            classes_ = ("iso8859-1",)

    no_unicode = decode_raw(NoUnicode(), raw)
    assert no_unicode.encoding == "iso8859-1"
    assert no_unicode.reason == "selected"
    print("UTF8_EVIDENCE_CONTRACT_PASS")

def thread_safety() -> None:
    model = CodecCat()
    raw_items = [
        b"hello",
        b"\xff\xfeh\x00i\x00",
        "日本語のテキスト".encode("utf-8"),
        ("café crème " * 200).encode("cp1252"),
        bytes(range(256)),
        b"",
        b"\x00" * 64,
    ]
    baseline = [model.detect(item) for item in raw_items]
    def worker(index: int):
        item = raw_items[index % len(raw_items)]
        return index % len(raw_items), model.detect(item)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(worker, range(1400)))
    assert all(result == baseline[i] for i, result in results)
    print("RELEASE_CONCURRENCY_PASS", json.dumps({"calls": len(results), "workers": 12}))


def cli_contract() -> None:
    root = pathlib.Path.cwd()
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "sample.bin"
        path.write_bytes("日本語".encode("utf-8"))
        commands = [
            [sys.executable, "-m", "codeccat", str(path), "--json"],
            [sys.executable, "-m", "codeccat", str(path), "--decode"],
            [sys.executable, "-m", "codeccat", "--help"],
        ]
        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, timeout=15, check=False)
            assert result.returncode == 0, (cmd, result.stderr.decode(errors="replace"))
        decoded = subprocess.run(
            commands[1], capture_output=True, timeout=15, check=True,
            env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        ).stdout
        assert decoded == "日本語".encode("utf-8"), decoded
        structured = json.loads(subprocess.run(
            commands[0], capture_output=True, timeout=15, check=True
        ).stdout.decode())
        assert structured["encoding"] is not None
        assert structured["accepted"] is True

        path.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
        oversized = subprocess.run(
            [sys.executable, "-m", "codeccat", str(path)],
            capture_output=True, timeout=15
        )
        assert oversized.returncode != 0, "CLI silently truncated oversized input"
    print("RELEASE_CLI_PASS")


if __name__ == "__main__":
    api_contract()
    utf8_evidence_contract()
    model_security()
    thread_safety()
    cli_contract()
    print("RELEASE_RELIABILITY_PASS", json.dumps({
        "python": sys.version.split()[0], "platform": sys.platform
    }))
