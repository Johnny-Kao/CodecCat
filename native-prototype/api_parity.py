"""Frozen public-API parity between original Alpha and isolated native Wheel.

Create reference in the sklearn training interpreter:
  PYTHONPATH=src:experiments/logic_first python native-prototype/api_parity.py generate
Verify in a fresh, zero-heavy-dependency native wheel interpreter:
  .native-clean/bin/python native-prototype/api_parity.py verify

No baseline imports or training dependencies in verify mode.
"""
from __future__ import annotations
import argparse,base64,hashlib,json
from pathlib import Path

OUTPUT=Path("native-prototype/api-parity-goldens.jsonl")
FP_GOLDEN=Path("native-prototype/api-model-fingerprint.txt")

def signature(r):
    return {
        "encoding":r.encoding,
        "text_sha256":hashlib.sha256(r.text.encode("utf-8","surrogatepass")).hexdigest()
                      if r.text is not None else None,
        "text_length":len(r.text) if r.text is not None else None,
        "accepted":r.accepted,
        "ambiguous":r.ambiguous,
        "candidates":list(r.candidates),
        "bom_encoding":r.bom_encoding,
        "bom_present":r.bom_present,
        "bom_policy":r.bom_policy,
        "reason":r.reason,
        "l3_called":r.l3_called,
        "l3_changed":r.l3_changed
    }

def cases():
    from vnext_actual_adapter import corpus
    rows=corpus(Path("training"))
    for raw,label,group,fold,sha in rows:
        yield raw
    yield from (
        b"",b"hello",b"a"*4096,b"hello\n\nworld",
        b"\xff\x80\x00\x81",bytes(range(256)),
        "日本語".encode("utf-8"),
        "日本語".encode("utf-16"),
        "日本語".encode("utf-32"),
        "\ufeffcafé".encode("utf-8"),
        "café".encode("cp1252"),
        b"\xef\xbb\xbf",
        b"\xff\xfe\x00\x00",
        b"\x00"*64,
        b"\x1b$B" + b"abc",
    )

def generate():
    from codeccat import CodecCat
    old=CodecCat.from_files("src/codeccat/models/stable23.zip")
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    FP_GOLDEN.write_text(old.model_fingerprint+"\n",encoding="ascii")
    l3=CodecCat.from_files("src/codeccat/models/stable23.zip",
                         language_path="src/codeccat/models/language-l3.json")
    n=0
    with OUTPUT.open("w",encoding="utf-8") as stream:
        for index,data in enumerate(cases()):
            for policy in ("strip","preserve"):
                for active,detector in ((False,old),(True,l3)) if index%12==0 else ((False,old),):
                    result=detector.detect(data,bom_policy=policy)
                    stream.write(json.dumps({
                        "raw":base64.b64encode(data).decode("ascii"),
                        "use_l3":active,
                        "expected":signature(result)
                    },ensure_ascii=False,separators=(",",":"))+"\n")
                    n+=1
    print("NATIVE_API_GOLDENS_GENERATED",n,flush=True)

def verify():
    import importlib.util
    from codeccat import CodecCat,detect,decode
    from codeccat._codeccat_native import NativeL2
    for name in ("numpy","sklearn","scipy","joblib"):
        assert importlib.util.find_spec(name) is None,name
    models={False:CodecCat(),True:CodecCat(use_l3=True)}
    expected_fp=FP_GOLDEN.read_text(encoding="ascii").strip()
    assert models[False].model_fingerprint==expected_fp, "public training fingerprint changed"
    n=0
    with OUTPUT.open(encoding="utf-8") as stream:
        for line in stream:
            record=json.loads(line)
            raw=base64.b64decode(record["raw"],validate=True)
            policy=record["expected"]["bom_policy"]
            result=models[record.get("use_l3",False)].detect(raw,bom_policy=policy)
            actual=signature(result)
            if actual!=record["expected"]:
                diff={k:(record["expected"][k],actual[k]) for k in actual
                      if actual[k]!=record["expected"][k]}
                raise AssertionError(f"API parity case {n}, bytes={len(raw)}: {diff}")
            n+=1
    assert n>=6000,n
    # Public one-call route with packaged default model, no manual file path.
    example=detect(b"hello")
    assert decode(b"hello")==example.text
    assert CodecCat.from_files("native-prototype/model/stable23.ccm").detect(b"hello")==example
    print("NATIVE_PUBLIC_API_PARITY_PASS",
          json.dumps({"cases":n,"numpy":False,"sklearn":False,
                      "model":"published Alpha stable23","l3":"enabled and disabled"}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("generate","verify"))
    args=p.parse_args()
    if args.mode=="generate":generate()
    else:verify()
