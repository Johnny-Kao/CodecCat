"""Paired external fixture diagnostic with unchanged stable23 + residual recipe.

Uses development training corpus to fit the same runtime model; external fixture
labels are filename-derived and NOT verified independent truth.
"""
import argparse, codecs, hashlib, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--training",required=True)
    p.add_argument("--external",required=True)
    p.add_argument("--out",default="research-results/external-labeled-diagnostic")
    args=p.parse_args()
    rows=corpus(Path(args.training))
    training=[(b,label,group) for b,label,group,_,_ in rows]
    residual,_=residual_subset(training)
    model=TrainableDetector.fit(residual,feature_profile="stable23")
    fixtures=[("iso8859-2","test/hu/iso-8859-2.txt"),("cp1250","test/hu/windows-1250.txt")]
    results=[]
    for label,relative in fixtures:
        data=(Path(args.external)/relative).read_bytes()
        truth=data.decode(label,"strict")
        answer=model.rank(data,final=len(data)<=4096)
        pred=answer.encoding
        try:pred_text=data.decode(pred,"strict") if pred else None
        except (UnicodeError,LookupError):pred_text=None
        results.append({"path":relative,"sha256":hashlib.sha256(data).hexdigest(),
                        "bytes":len(data),"declared_label":label,"predicted":pred,
                        "label_match":pred==label,"strict_same_text":pred_text==truth,
                        "candidates":answer.candidates,"ambiguous_flag":answer.ambiguous})
    report={"schema":"codeccat-third-party-labeled-diagnostic-v1",
            "status":"development-external-fixtures-not-blind-acceptance",
            "training_count":len(residual),"test_count":len(results),
            "strict_same_text_count":sum(x["strict_same_text"] for x in results),
            "label_match_count":sum(x["label_match"] for x in results),
            "cases":results,
            "limits":["Only two labeled fixture documents, not statistically representative.",
                      "Labels inferred from fixture filenames, not independently verified.",
                      "Training originates from chardet/test-data and may overlap external fixture lineage.",
                      "The runtime weights are fitted for this diagnostic, not the released serialized model."]}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("EXTERNAL_LABELED_DIAGNOSTIC_START")
    print(json.dumps(report,indent=2))
    print("EXTERNAL_LABELED_DIAGNOSTIC_END")
if __name__=="__main__":
    main()
