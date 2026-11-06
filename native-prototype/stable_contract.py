"""Stable native API invariants, executed against freshly installed release wheel.

This is a release contract check, not an encoding-accuracy benchmark.
"""
from __future__ import annotations
import dataclasses
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import codeccat

FIELDS=("encoding","text","candidates","bom_encoding","bom_present","ambiguous",
        "reason","bom_policy","l3_called","l3_changed")
assert dataclasses.is_dataclass(codeccat.Detection)
assert all(f in codeccat.Detection.__dataclass_fields__ for f in FIELDS)
assert codeccat.Detection.__dataclass_params__.frozen
assert {"CodecCat","Detection","decode","detect"} <= set(codeccat.__all__)
assert all(importlib.util.find_spec(x) is None for x in ("numpy","scipy","sklearn","joblib"))

for inp in (b"hello", bytearray(b"hello"), memoryview(b"hello")):
    r=codeccat.detect(inp)
    assert r.text=="hello" and r.accepted
    assert isinstance(r.candidates,tuple)
    assert isinstance(r.ambiguous,bool)
    assert isinstance(r.reason,str)
    assert codeccat.decode(inp)=="hello"
    assert codeccat.CodecCat().detect(inp).text=="hello"

for invalid in (None,"hello",1,{"x":"y"}):
    try:codeccat.detect(invalid)
    except TypeError:pass
    else:raise AssertionError("accepted non-bytes input")

try:codeccat.detect(b"hello",bom_policy="bad")
except ValueError:pass
else:raise AssertionError("accepted invalid BOM policy")

payload=b"\xef\xbb\xbfhello"
assert codeccat.decode(payload,bom_policy="strip")=="hello"
assert codeccat.decode(payload,bom_policy="preserve")=="\ufeffhello"

for value in (b"hello",payload,b"",bytes(range(256)),"日本語".encode("utf-8")):
    assert codeccat.detect(value)==codeccat.detect(value)
    assert codeccat.CodecCat().detect(value)==codeccat.detect(value)

with tempfile.TemporaryDirectory() as d:
    p=Path(d)/"in.bin"
    p.write_bytes(b"\xef\xbb\xbf"+"café".encode("utf-8"))
    cmd=[sys.executable,"-m","codeccat",str(p),"--json"]
    run=subprocess.run(cmd,capture_output=True,timeout=30,check=True)
    data=json.loads(run.stdout)
    assert {"encoding","accepted","ambiguous","candidates","bom_present","reason"} <= set(data)
    cmd=[sys.executable,"-m","codeccat",str(p),"--decode"]
    run=subprocess.run(cmd,capture_output=True,timeout=30,check=True)
    assert run.stdout.decode("utf-8")=="café"
    p.write_bytes(b"x"*(16*1024*1024+1))
    run=subprocess.run([sys.executable,"-m","codeccat",str(p)],capture_output=True,timeout=30)
    assert run.returncode!=0 and b"limit" in run.stderr

print("STABLE_PUBLIC_API_CONTRACT_PASS",importlib.metadata.version("codeccat"))
