"""Acceptance of an actually usable installed alpha wheel, no PYTHONPATH."""
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from importlib.resources import files

import codeccat


def main():
    root = Path(codeccat.__file__).resolve()
    assert "site-packages" in str(root), root
    assert importlib.metadata.version("codeccat") == "0.1.0a1"
    resources = files("codeccat")
    for name in ("stable23.zip", "language-l3.json", "manifest.json"):
        assert resources.joinpath("models", name).is_file(), name
    model_meta = json.loads(resources.joinpath("models", "manifest.json").read_text())
    assert model_meta["schema"] == "codeccat-alpha-bundled-model-v1"
    assert model_meta["classes"] >= 40

    raw = b"\xef\xbb\xbfHello, CodecCat!"
    result = codeccat.detect(raw)
    assert result.accepted, result
    assert result.text == "Hello, CodecCat!", result
    assert result.bom_present
    assert codeccat.decode(raw) == result.text
    assert codeccat.decode(bytearray(raw)) == result.text
    assert codeccat.decode(memoryview(raw)) == result.text
    assert codeccat.detect(raw, bom_policy="preserve").text == "\ufeffHello, CodecCat!"

    instance = codeccat.CodecCat()
    assert instance.model_fingerprint == model_meta["model"]["fingerprint"]
    assert instance.detect(raw).text == result.text
    assert instance.decode(raw) == result.text
    l3_instance = codeccat.CodecCat(use_l3=True)
    assert l3_instance.detect(raw).text == result.text

    utf16 = "Sample in UTF-16".encode("utf-16")
    decoded_utf16 = codeccat.detect(utf16)
    assert decoded_utf16.accepted and decoded_utf16.text == "Sample in UTF-16", decoded_utf16

    ascii_result = codeccat.detect(b"A plain ASCII message.")
    assert ascii_result.accepted and ascii_result.text == "A plain ASCII message."
    assert ascii_result.ambiguous
    try:
        codeccat.detect("already decoded unicode")
        raise AssertionError("text input must fail")
    except TypeError:
        pass

    cli = subprocess.run(
        [sys.executable, "-m", "codeccat", "--json"],
        input=raw, capture_output=True, check=True,
    )
    obj = json.loads(cli.stdout)
    assert obj["accepted"] and obj["bom_present"], obj
    strict = subprocess.run(
        [sys.executable, "-m", "codeccat", "--decode"],
        input=raw, capture_output=True, check=True,
    )
    assert strict.stdout == b"Hello, CodecCat!", strict

    print(json.dumps({
        "status": "USABLE_ALPHA_PASS",
        "version": importlib.metadata.version("codeccat"),
        "loaded_model": model_meta["model"]["fingerprint"],
        "trained_classes": model_meta["classes"],
        "default_api": "codeccat.detect(codeccat_bytes)",
        "text_api": "codeccat.decode(codeccat_bytes)",
        "cli": "python -m codeccat --json / --decode",
        "bom": "PASS", "ascii": "PASS", "utf16": "PASS",
        "optional_l3": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
