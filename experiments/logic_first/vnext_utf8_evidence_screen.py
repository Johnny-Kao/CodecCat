"""Safety screen for a structural UTF-8 rescue gate; no production mutation.

A code-point count threshold is a proposed DOMAIN-INDEPENDENT criterion,
not a language-specific patch. Compare against source-group/content-disjoint
development folds and already-consumed independent Vim PO windows.
This is NOT fresh independent acceptance evidence.
"""
from __future__ import annotations

import argparse,collections,json,pathlib,sys,unicodedata

from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_bom import inspect_bom
from codeccat.vnext_decoding import decode
from codeccat.vnext_utf8_evidence import Utf8EvidenceDetector

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from vnext_training_strategy_audit import strict_text,family
from vnext_content_safe_contrast import component_folds
from vnext_vim_catalog_one_shot import get_frozen,verify_commit,VIM_REV,TRAIN_REV

THRESHOLDS=(8,16,32,64,128)


def unicode_candidate(raw):
    if len(raw)<64 or inspect_bom(raw,final=True).signature:
        return None,0
    try: text=raw.decode("utf-8","strict")
    except UnicodeDecodeError:return None,0
    exotic=sum(ord(c)>=128 for c in text)
    controls=sum(unicodedata.category(c)=="Cc" and c not in ("\t","\n","\r")
                 for c in text)
    if controls:return None,0
    return text,exotic


def report(cases):
    n=len(cases)
    outcome={"n":n,"base_text_correct":sum(c["base_correct"] for c in cases)}
    for k in THRESHOLDS:
        h=sum(c["non_ascii_codepoints"]>=k and c["utf8_valid"] and
              not c["base_correct"] and c["utf8_correct"] and
              c["baseline_label"]!="utf-8" for c in cases)
        d=sum(c["non_ascii_codepoints"]>=k and c["utf8_valid"] and
              c["base_correct"] and not c["utf8_correct"] and
              c["baseline_label"]!="utf-8" for c in cases)
        changes=sum(c["non_ascii_codepoints"]>=k and c["utf8_valid"] and
                    c["baseline_label"]!="utf-8" for c in cases)
        harming=[c["truth_label"] for c in cases if c["non_ascii_codepoints"]>=k and
                  c["utf8_valid"] and c["base_correct"] and not c["utf8_correct"] and
                  c["baseline_label"]!="utf-8"]
        outcome[str(k)]={"helped":h,"harmed":d,"net":h-d,"changed":changes,
                        "harm_true_encodings":dict(collections.Counter(harming))}
    return outcome


def source_cases(args):
    from vnext_vim_catalog_one_shot import get_frozen
    rows=get_frozen(args.source,args.manifest)
    raw=[]
    seen=set()
    for item in rows:
        if item["sha"] in seen:continue
        seen.add(item["sha"])
        raw.append(item)
    return raw


def run(args):
    verify_commit(args.source,VIM_REV);verify_commit(args.training,TRAIN_REV)
    rows=corpus(pathlib.Path(args.training))
    assignment,split=component_folds(rows)
    folds=[]
    for fold in range(3):
        train=[(b,y,g) for (b,y,g,_,digest),f in zip(rows,assignment) if f!=fold]
        test=[(b,y,g,digest) for (b,y,g,_,digest),f in zip(rows,assignment) if f==fold]
        residual,_=residual_subset(train)
        model=TrainableDetector.fit(residual)
        verified=Utf8EvidenceDetector(model,min_codepoints=16)
        evaluations=[]
        for raw,label,group,digest in test:
            truth=strict_text(raw,label)
            ranked=model.rank(raw,final=len(raw)<4096)
            original=strict_text(raw,ranked.encoding)
            rescued=verified.rank(raw,final=len(raw)<4096)
            rescued_text=strict_text(raw,rescued.encoding)
            predicted,nonascii=unicode_candidate(raw)
            evaluations.append({
                "truth_label":label,"baseline_label":str(ranked.encoding),
                "base_correct":original==truth and original is not None,
                "actual_rescue_correct":rescued_text==truth and rescued_text is not None,
                "actual_rescue_encoding":str(rescued.encoding),
                "utf8_valid":predicted is not None,
                "utf8_correct":predicted==truth and predicted is not None,
                "non_ascii_codepoints":nonascii,
            })
        r=report(evaluations)
        r["actual_rescue_correct"]=sum(x["actual_rescue_correct"] for x in evaluations)
        assert r["actual_rescue_correct"]-r["base_text_correct"]==r["16"]["net"], "actual adapter differs from structural risk screen"
        folds.append({"fold":fold,**r})
        print("UTF8_EVIDENCE_FOLD",fold,json.dumps({k:r[k] for k in map(str,THRESHOLDS)}),flush=True)
    complete=[(raw,y,g) for raw,y,g,fold,digest in rows]
    focused,_=residual_subset(complete)
    trained=TrainableDetector.fit(focused)
    verified=Utf8EvidenceDetector(trained,min_codepoints=16)
    vim_cases=source_cases(args)
    vim_evaluations=[]
    for item in vim_cases:
        raw=item["raw"];answer=decode(trained,raw,bom_policy="preserve")
        actual=decode(verified,raw,bom_policy="preserve")
        proposed,nonascii=unicode_candidate(raw)
        vim_evaluations.append({
            "file":item["file"],"truth_label":item["codec"],
            "baseline_label":answer.encoding,"base_correct":answer.text==item["text"],
            "actual_rescue_correct":actual.text==item["text"],
            "actual_rescue_encoding":actual.encoding,
            "utf8_valid":proposed is not None,
            "utf8_correct":proposed==item["text"] and proposed is not None,
            "non_ascii_codepoints":nonascii,
        })
    vim=report(vim_evaluations)
    vim["actual_rescue_correct"]=sum(x["actual_rescue_correct"] for x in vim_evaluations)
    assert vim["actual_rescue_correct"]-vim["base_text_correct"]==vim["16"]["net"], "actual adapter disagrees on Vim source"
    agg={"n":sum(f["n"] for f in folds)}
    for k in THRESHOLDS:
        agg[str(k)]={x:sum(f[str(k)][x] for f in folds)
                     for x in ("helped","harmed","net","changed")}
    result={
        "schema":"codeccat-utf8-evidence-risk-screen-v1",
        "status":"DEVELOPMENT_SCREEN_CONSUMED_HOLDOUT_ONLY",
        "assumption":"Strict valid UTF-8 with >=threshold non-ASCII Unicode codepoints; no unexpected controls or BOM",
        "source_fold_split":split,
        "folds":folds,"aggregate_development":agg,"consumed_vim_original_bytes":vim,
        "limits":[
            "If positive, requires a NEW unseen source before promoting any rule",
            "UTF-8 validity alone does not identify historical encoding provenance",
            "Partially observed 4096-byte strings are not used unless strict UTF-8 decodes whole prefix",
        ],
    }
    dest=pathlib.Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/"report.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("UTF8_EVIDENCE_RISK_SCREEN_RESULT_BEGIN")
    print(json.dumps({"dev":agg,"vim":vim},sort_keys=True,indent=2))
    print("UTF8_EVIDENCE_RISK_SCREEN_RESULT_END")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--training",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",default="research-results/utf8-validity-screen")
    run(p.parse_args())
