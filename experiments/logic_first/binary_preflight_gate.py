"""Source-disjoint acceptance audit: text recall vs synthetic binary rejection.

This tests binary preflight in isolation and leaves original model unchanged.
"""
import hashlib,json,collections,statistics,time
from pathlib import Path
from vnext_actual_adapter import corpus
from semantic_stability_audit import negatives
from enca_external_gate import ALIASES
from l2_variant_matrix import valid
from codeccat.vnext_binary_preflight import inspect_text

def main():
    rows=corpus(Path("training"))
    counts=collections.Counter();failures=[];times=[]
    for raw,enc,group,fold,sha in rows:
        t=time.process_time_ns()
        verdict=inspect_text(raw)
        times.append((time.process_time_ns()-t)/1e6)
        counts["development_text"]+=1
        counts["development_accept"]+=int(verdict.text_possible)
        if not verdict.text_possible and len(failures)<30:
            failures.append({"source":"development","label":enc,"group":group,"reason":verdict.reason,"sha":sha})
    exact={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
    prefixes={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
    for p in sorted(Path("external/enca/test").iterdir()):
        if not p.is_file() or len(p.name.split("."))!=2:continue
        enc=ALIASES.get(p.name.split(".")[-1])
        if enc is None:continue
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest() in exact or hashlib.sha256(raw[:4096]).hexdigest() in prefixes:continue
        if valid(raw,enc) is None:continue
        verdict=inspect_text(raw)
        counts["enca_text"]+=1
        counts["enca_accept"]+=int(verdict.text_possible)
        if not verdict.text_possible and len(failures)<50:
            failures.append({"source":"enca","label":enc,"file":p.name,"reason":verdict.reason})
    reasons=collections.Counter()
    for category,raw in negatives():
        verdict=inspect_text(raw)
        counts["synthetic_negative"]+=1
        counts["synthetic_rejected"]+=int(not verdict.text_possible)
        reasons[(category,verdict.reason)]+=1
    # Preserve PDF-like input by default: PDF can contain literal text.
    out={"counts":dict(counts),"text_false_rejections":failures,
         "negative_reasons":[{"category":cat,"reason":reason,"n":n} for (cat,reason),n in reasons.most_common()],
         "cpu_p50_ms":statistics.median(times),
         "cpu_p95_ms":sorted(times)[int((len(times)-1)*.95)],
         "notes":"Synthetic negative cases are NOT real-world independent binary documents. PDF can be text. Excluding exact and 4096-prefix overlaps cannot remove same text under re-encoding. Do not auto-promote."}
    root=Path("research-results/binary-preflight")
    root.mkdir(parents=True,exist_ok=True)
    (root/"report.json").write_text(json.dumps(out,indent=2))
    print("BINARY_PREFLIGHT_RESULT",json.dumps({k:v for k,v in out.items() if k not in ("text_false_rejections","negative_reasons")}),flush=True)
    print("BINARY_PREFLIGHT_FALSE_REJECTIONS",json.dumps(failures[:20]),flush=True)
    print("BINARY_PREFLIGHT_NEGATIVE_REASONS",json.dumps(out["negative_reasons"]),flush=True)
if __name__=="__main__":main()
