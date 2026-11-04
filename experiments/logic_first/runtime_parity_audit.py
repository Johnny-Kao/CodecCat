"""Paired historical decision vs complete-document runtime parity diagnostic.

Read-only evaluation. Shared training/folds; no model or threshold mutation.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json
import codecs
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from l2_variant_matrix import valid
from codeccat.vnext_declarations import choose
from codeccat.vnext_l3_language import LanguageCoherence
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_binary_preflight import inspect_text
from codeccat.vnext_decoding import _literal_decode

def canonical(raw, label):
    try:
        return codecs.getincrementaldecoder(label)("strict").decode(raw, final=True).lstrip("\ufeff")
    except (UnicodeError,LookupError,ValueError,TypeError):
        return None

def main():
    rows=corpus(Path("training"))
    counts=Counter()
    reasons=defaultdict(Counter)
    examples=[]
    for fold in range(3):
        train=[(b,y,g) for b,y,g,k,sha in rows if k!=fold]
        model=fit(residual_subset(train)[0],"binary_high")
        language=LanguageCoherence.fit(train)
        for raw,label,group,k,sha in rows:
            if k!=fold:continue
            old_final=len(raw)<4096
            legacy_rank=model.rank(raw,final=old_final)
            complete_rank=model.rank(raw,final=True)
            decl=choose(raw,legacy_rank.encoding)
            l3=language.rerank(legacy_rank,raw,final=old_final,max_l2_margin=.5,min_language_advantage=.15)
            historic=decl.encoding if decl.reason.startswith("declared-") else l3.encoding
            runtime=detect(model,raw,language_model=language)
            expected=valid(raw,label)
            baseline_ok=valid(raw,legacy_rank.encoding)==expected
            historic_ok=valid(raw,historic)==expected
            runtime_choice_ok=valid(raw,runtime.encoding)==expected if runtime.accepted else False
            runtime_text_ok=runtime.text==expected and runtime.accepted
            strict=canonical(raw,label)
            counts["n"]+=1
            counts["length_4096"]+=int(len(raw)==4096)
            counts["baseline_ok"]+=int(baseline_ok)
            counts["historic_ok"]+=int(historic_ok)
            counts["runtime_selected_ok_prefix"]+=int(runtime_choice_ok)
            counts["runtime_decoded_ok_prefix"]+=int(runtime_text_ok)
            counts["strict_truth_valid"]+=int(strict is not None)
            counts["strict_runtime_ok"]+=int(strict is not None and runtime.text==strict and runtime.accepted)
            counts["rank_changed"]+=int(legacy_rank.encoding!=complete_rank.encoding)
            counts["final_choice_changed"]+=int(historic!=runtime.encoding)
            counts["historic_pass_runtime_fail"]+=int(historic_ok and not runtime_choice_ok)
            counts["historic_fail_runtime_pass"]+=int(not historic_ok and runtime_choice_ok)
            counts["prefix_decoder_disagreement"]+=int(runtime_choice_ok and not runtime_text_ok)
            if historic_ok and not runtime_choice_ok:
                tags=[]
                if legacy_rank.encoding!=complete_rank.encoding:tags.append("rank_finality")
                if not inspect_text(raw).text_possible:tags.append("preflight")
                if runtime.reason=="invalid-selected-codec":tags.append("strict_decode_reject")
                if historic!=runtime.encoding:tags.append("decision_changed")
                if not tags:tags.append("other")
                for tag in tags:reasons["historic_to_runtime_regression"][tag]+=1
                if len(examples)<80:examples.append({"sha":sha,"fold":fold,"bytes":len(raw),"truth":label,"old_rank":legacy_rank.encoding,"complete_rank":complete_rank.encoding,"historic":historic,"runtime":runtime.encoding,"runtime_reason":runtime.reason,"tags":tags})
            if not historic_ok and runtime_choice_ok:
                reasons["runtime_improvements"]["rank_changed" if legacy_rank.encoding!=complete_rank.encoding else "same_rank"]+=1
    report={"counts":dict(counts),"reasons":{k:dict(v) for k,v in reasons.items()},"examples":examples,
    "note":"Historical valid() uses non-final 4096-byte prefix; runtime strict decoding requires final=True. Scores on the same folds and records. Do not treat prefix and strict accuracies as interchangeable."}
    p=Path("research-results/runtime-parity");p.mkdir(parents=True,exist_ok=True)
    (p/"report.json").write_text(json.dumps(report,indent=2))
    print("RUNTIME_PARITY_SUMMARY",json.dumps({k:v for k,v in report.items() if k!="examples"}),flush=True)
if __name__=="__main__":main()
