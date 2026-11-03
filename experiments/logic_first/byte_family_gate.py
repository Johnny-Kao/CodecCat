"""Grouped hold-out byte-family proposals vs fixed primary SGD; Enca isolated.
Measures candidate *coverage* before implementing any potentially harmful override.
"""
from collections import Counter
from pathlib import Path
import hashlib,codecs,json,statistics,time
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from l2_variant_matrix import valid
from enca_external_gate import ALIASES
from codeccat.vnext_byte_families import ByteFamilies,candidate_union

def run(rows):
    output={}
    for fold in range(4):
        if fold<3:
            train=[(b,y,g) for b,y,g,k,sha in rows if k!=fold]
            test=[(b,y,g,sha) for b,y,g,k,sha in rows if k==fold]
            source="development"
        else:
            train=[(b,y,g) for b,y,g,k,sha in rows]
            known={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
            pref={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
            test=[]
            for path in sorted(Path("external/enca/test").iterdir()):
                if not path.is_file() or len(path.name.split("."))!=2:continue
                label=ALIASES.get(path.name.split(".")[-1])
                if label is None:continue
                try:codecs.lookup(label)
                except LookupError:continue
                raw=path.read_bytes()
                if hashlib.sha256(raw).hexdigest() in known or hashlib.sha256(raw[:4096]).hexdigest() in pref:continue
                if valid(raw,label) is None:continue
                test.append((raw,label,path.name,path.name))
            source="enca"
        model=fit(residual_subset(train)[0],"binary_high")
        family=ByteFamilies.fit(train)
        measures=output.setdefault(source,{str(k):Counter() for k in (3,8,16)})
        latency=[]
        examples=[]
        for raw,label,group,sha in test:
            truth=valid(raw,label)
            pred=model.rank(raw,final=len(raw)<4096)
            base_correct=valid(raw,pred.encoding)==truth
            start=time.process_time_ns()
            proposals=family.proposals(raw,limit=16)
            elapsed=(time.process_time_ns()-start)/1e6
            latency.append(elapsed)
            for num in (3,8,16):
                key=str(num)
                # Include original Top3; cap family proposal additions to num.
                names=list(pred.candidates)
                if not pred.bom_hint and len(pred.candidates)>1:
                    for p in proposals[:num]:
                        if p not in names and family_safe_valid(raw,p):
                            names.append(p)
                reachable=any(valid(raw,name)==truth for name in names)
                m=measures[key]
                m["n"]+=1;m["base_correct"]+=int(base_correct)
                m["base_recall"]+=int(any(valid(raw,n)==truth for n in pred.candidates))
                m["union_recall"]+=int(reachable)
                m["newly_recoverable"]+=int(not any(valid(raw,n)==truth for n in pred.candidates) and reachable)
                m["union_size_total"]+=len(names)
            if source=="enca" and not base_correct and len(examples)<36:
                examples.append({"name":sha,"truth":label,"pred":pred.encoding,"proposals":list(proposals[:8])})
        extra=output.setdefault(source+"_perf",[])
        extra.append({"fold":fold,"n":len(test),"candidate_scoring_cpu_p50_ms":statistics.median(latency) if latency else None,"candidate_scoring_cpu_p95_ms":sorted(latency)[int((len(latency)-1)*.95)] if latency else None})
        if source=="enca":output["enca_examples"]=examples
    summary={key:({k:dict(v) for k,v in value.items()} if isinstance(value,dict) else value) for key,value in output.items()}
    return summary

def family_safe_valid(raw,label):
    try:
        import codecs
        dec=codecs.getincrementaldecoder(label)("strict")
        dec.decode(raw[:4096],final=len(raw)<4096)
        return True
    except (UnicodeError,LookupError,ValueError):return False

def main():
    out=run(corpus(Path("training")))
    dest=Path("research-results/byte-family-proposal")
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"report.json").write_text(json.dumps(out,indent=2))
    print("BYTE_FAMILY_PROPOSAL_RESULT",json.dumps({k:v for k,v in out.items() if k!="enca_examples"}),flush=True)
if __name__=="__main__":main()
