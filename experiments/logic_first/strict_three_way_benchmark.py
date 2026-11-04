"""Strict complete-document paired benchmark on frozen 3003-eligible group folds.

Same raw bytes and Unicode decoded-text equality for all detectors. Development
diagnostic, not blind release evidence; chardet owns the corpus.
"""
from pathlib import Path
from collections import Counter, defaultdict
import codecs, hashlib, json, platform, statistics, time
import chardet, charset_normalizer
from charset_normalizer import from_bytes
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence

def strict(raw,encoding):
    if not encoding:return None
    try:
        return codecs.getincrementaldecoder(encoding)("strict").decode(raw, final=True).lstrip("\ufeff")
    except (UnicodeError,LookupError,ValueError,TypeError):return None

def main():
    rows=corpus(Path("training"))
    n=0; counters=defaultdict(Counter); latency=defaultdict(list)
    cases=[]
    for fold in range(3):
        train=[(b,y,g) for b,y,g,k,sha in rows if k!=fold]
        model=fit(residual_subset(train)[0],"binary_high")
        language=LanguageCoherence.fit(train)
        for raw,label,group,k,sha in rows:
            if fold!=k:continue
            expected=strict(raw,label)
            if expected is None:continue
            n+=1
            t=time.process_time_ns(); r=detect(model,raw,language_model=language)
            latency["codeccat"].append((time.process_time_ns()-t)/1e6)
            t=time.process_time_ns(); cd=chardet.detect(raw).get("encoding")
            latency["chardet"].append((time.process_time_ns()-t)/1e6)
            t=time.process_time_ns(); best=from_bytes(raw).best()
            latency["charset-normalizer"].append((time.process_time_ns()-t)/1e6)
            cn=best.encoding if best else None
            predictions={"codeccat":r.text if r.accepted else None,
                         "chardet":strict(raw,cd),"charset-normalizer":strict(raw,cn)}
            entry={"sha":sha,"fold":fold,"truth":label,"group":group,"bytes":len(raw),
                   "predicted":{"codeccat":r.encoding,"chardet":cd,"charset-normalizer":cn},
                   "correct":{}}
            for key,decoded in predictions.items():
                ok=decoded is not None and decoded==expected
                counters[key]["n"]+=1;counters[key]["correct"]+=int(ok)
                entry["correct"][key]=ok
            cases.append(entry)
        print("STRICT_PAIRED_FOLD_DONE",fold,flush=True)
    assert n==3003,(n,"strict eligibility unexpectedly shifted")
    report={"n":n,"evaluation":"identical full-document strict decoded Unicode equality, leading FEFF stripped",
        "accuracy":{k:{**dict(v),"percentage":round(100*v["correct"]/v["n"],3),
            "cpu_p50_ms":round(statistics.median(latency[k]),4)} for k,v in counters.items()},
        "versions":{"chardet":chardet.__version__,"charset-normalizer":charset_normalizer.__version__,
                    "python":platform.python_version()},
        "paired":{"cc_only_vs_chardet":sum(c["correct"]["codeccat"] and not c["correct"]["chardet"] for c in cases),
                  "chardet_only_vs_cc":sum(c["correct"]["chardet"] and not c["correct"]["codeccat"] for c in cases),
                  "cc_only_vs_cn":sum(c["correct"]["codeccat"] and not c["correct"]["charset-normalizer"] for c in cases),
                  "cn_only_vs_cc":sum(c["correct"]["charset-normalizer"] and not c["correct"]["codeccat"] for c in cases)},
        "limitations":["chardet owns the evaluation corpus; potential source advantage",
          "frozen fold grouping but development corpus used to choose architecture",
          "CPU time on shared runner is indicative, not steady-state performance benchmark"]}
    out=Path("research-results/strict-three-way");out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases)+"\n")
    print("STRICT_THREE_WAY_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
