"""Seeded stress audit: same fold-trained runtime on original and valid byte-prefix variants.

No change to model or inference policy. Variants are *new input tasks*, not
semantics-preserving transformations. Accuracy is evaluated against the
expected Unicode text for each variant, not the original text.
"""
from pathlib import Path
from collections import Counter, defaultdict
import codecs,hashlib,json,random,statistics,sys
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence

SEED=20261105

def strict(raw,enc):
    try:
        return codecs.getincrementaldecoder(enc)("strict").decode(raw,final=True).lstrip("\ufeff")
    except (UnicodeError,LookupError,TypeError,ValueError):return None

def attempt_prefix(raw,enc,target):
    # Retain the original byte offset, then retreat to a strict-valid boundary.
    # Exclude very short/empty results and avoid changing the initial signature.
    end=min(target,len(raw)-1)
    for delta in range(9):
        if end-delta<64:break
        candidate=raw[:end-delta]
        if strict(candidate,enc) is not None:return candidate
    return None

def main():
    rows=corpus(Path("training"))
    rng=random.Random(SEED)
    grouped=defaultdict(list)
    for item in rows:
        raw,enc,group,fold,sha=item
        if strict(raw,enc) is not None:grouped[(fold,enc)].append(item)
    # Balance per-class selection; do not pretend it is a new independent corpus.
    chosen=set()
    for key,bucket in sorted(grouped.items()):
        rng.shuffle(bucket)
        for item in bucket[:min(8,len(bucket))]:
            chosen.add(item[4])
    summary=defaultdict(Counter)
    paired=defaultdict(Counter)
    cases=[]
    for fold in range(3):
        train=[(b,y,g) for b,y,g,k,sha in rows if k!=fold]
        model=fit(residual_subset(train)[0],"binary_high")
        language=LanguageCoherence.fit(train)
        testing=[(b,y,g,sha) for b,y,g,k,sha in rows if k==fold and sha in chosen]
        rng.shuffle(testing)
        for raw,enc,group,sha in testing:
            variants={"original":raw}
            # Distinct byte budgets probe shift in model features; strict-final
            # validity is checked for every candidate and its own ground truth.
            for target in (128,512,1024,2048):
                if len(raw)<=target+32:continue
                prefix=attempt_prefix(raw,enc,target)
                if prefix is not None:variants["prefix_"+str(target)]=prefix
            original_correct=None
            for kind,variant in variants.items():
                expected=strict(variant,enc)
                if expected is None:continue
                result=detect(model,variant,language_model=language)
                correct=result.accepted and result.text==expected
                c=summary[kind];c["n"]+=1;c["correct"]+=int(correct)
                c["rejected"]+=int(not result.accepted)
                if kind=="original":original_correct=correct
                elif original_correct is not None:
                    p=paired[kind];p["n"]+=1
                    p["original_correct_variant_wrong"]+=int(original_correct and not correct)
                    p["original_wrong_variant_correct"]+=int(not original_correct and correct)
                cases.append({"sha":sha,"source_group":group,"fold":fold,
                    "truth":enc,"variant":kind,"bytes":len(variant),
                    "chosen":result.encoding,"correct":correct,"rejected":not result.accepted,
                    "original_correct":original_correct})
        print("STRESS_FOLD_DONE",fold,flush=True)
    report={"seed":SEED,"selected_base_documents":len(chosen),
        "by_variant":{k:{**dict(v),"accuracy":round(v["correct"]/v["n"],5)} for k,v in summary.items()},
        "paired":{k:dict(v) for k,v in paired.items()},
        "limitations":["Same development source as training; source-group folds prevent within-fold overlap only.",
          "Prefixes are independently scored inputs, not expected to retain same predicted encoding.",
          "Truncation validity is checked with strict final decoding. Stateful encodings may be omitted.",
          "Results are diagnostic sensitivity checks, not independent generalization estimates."]}
    p=Path("research-results/seeded-robustness");p.mkdir(parents=True,exist_ok=True)
    (p/"summary.json").write_text(json.dumps(report,indent=2))
    (p/"cases.jsonl").write_text("\n".join(json.dumps(x) for x in cases)+"\n")
    print("SEEDED_ROBUSTNESS_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
