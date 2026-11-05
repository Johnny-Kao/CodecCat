"""P1 numeric precision experiment: compare f32 CCM2 to f64 Alpha goldens.

Independent evaluation only; no threshold tuning. Runs in clean no-ML venv.
"""
import argparse,base64,json,hashlib
from pathlib import Path
from codeccat import CodecCat
from codeccat.vnext_decoding import Detection
from codeccat._codeccat_native import NativeL2
from codeccat.vnext_l3_language import LanguageCoherence

GOLDENS=Path("native-prototype/api-parity-goldens.jsonl")
F32=Path("native-prototype/model/stable23-f32.ccm")
REPORT=Path("native-prototype/precision-results.json")

def signature(r):
    return {
        "encoding":r.encoding,
        "text_sha256":hashlib.sha256(r.text.encode("utf-8","surrogatepass")).hexdigest()
          if r.text is not None else None,
        "text_length":len(r.text) if r.text is not None else None,
        "accepted":r.accepted,"ambiguous":r.ambiguous,
        "candidates":list(r.candidates),
        "bom_encoding":r.bom_encoding,"bom_present":r.bom_present,
        "bom_policy":r.bom_policy,"reason":r.reason,
        "l3_called":r.l3_called,"l3_changed":r.l3_changed,
    }

def main():
    model=NativeL2.from_model_file(str(F32))
    assert model.numeric_format()=="f32"
    candidates={
        False:CodecCat.from_files(F32),
        True:CodecCat.from_files(F32,language_path="native-prototype/python/codeccat/models/language-l3.json"),
    }
    diffs=[];n=0;fields={}
    with GOLDENS.open(encoding="utf-8") as stream:
        for line in stream:
            case=json.loads(line)
            raw=base64.b64decode(case["raw"],validate=True)
            policy=case["expected"]["bom_policy"]
            actual=signature(candidates[bool(case.get("use_l3",False))].detect(raw,bom_policy=policy))
            changed={k:[case["expected"][k],v] for k,v in actual.items()
                     if case["expected"][k]!=v}
            if changed:
                for key in changed:fields[key]=fields.get(key,0)+1
                diffs.append({"index":n,"bytes":len(raw),"different":changed})
            n+=1
    report={"cases":n,"different_cases":len(diffs),"changed_fields":fields,
        "f32_model_bytes":F32.stat().st_size,
        "f64_model_bytes":Path("native-prototype/model/stable23.ccm").stat().st_size,
        "samples":diffs[:30],
        "accept_for_default":len(diffs)==0,
        "note":"Pass requires complete exact detection API parity, not merely same accuracy count. Never tune on this development fixture."}
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("NATIVE_F32_EXPERIMENT",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
