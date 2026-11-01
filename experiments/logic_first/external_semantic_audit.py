"""Follow-up: content-transcode leakage audit and case-level failure attribution."""
import argparse,codecs,hashlib,json,sys
from collections import defaultdict,Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
from external_accuracy_tournament import truth_label,decode_prefix,canonical,measure

def normalize(text):
    return hashlib.sha256(text.replace("\r\n","\n").replace("\r","\n").lstrip("\ufeff").encode("utf-8")).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--training",required=True);p.add_argument("--external",required=True)
    p.add_argument("--out",default="research-results/semantic-leakage")
    a=p.parse_args()
    rows=corpus(Path(a.training))
    texts=defaultdict(list)
    for b,label,group,fold,digest in rows:
        decoded=decode_prefix(b,label)
        if decoded is not None:texts[normalize(decoded)].append((label,group,digest))
    training=[(b,y,g) for b,y,g,_,_ in rows]
    residual,_=residual_subset(training)
    model=TrainableDetector.fit(residual,feature_profile="stable23")
    import chardet,charset_normalizer
    cases=[]; outgroups=defaultdict(list)
    for path in sorted((Path(a.external)/"test").rglob("*")):
        if not path.is_file():continue
        label=truth_label(path)
        if not label:continue
        raw=path.read_bytes()
        truth=decode_prefix(raw,label)
        if truth is None:continue
        text_hash=normalize(truth)
        rank=model.rank(raw,final=len(raw)<=4096)
        pred=canonical(str(rank.encoding)) if rank.encoding else None
        try:c=chardet.detect(raw);ch=canonical(c["encoding"]) if c["encoding"] else None
        except Exception:ch=None
        try:best=charset_normalizer.from_bytes(raw).best();cn=canonical(best.encoding) if best else None
        except Exception:cn=None
        item={"path":path.relative_to(a.external).as_posix(),"label":label,"prediction":pred,
              "top3":list(rank.candidates),"sha256":hashlib.sha256(raw).hexdigest(),
              "text_hash":text_hash,"identical_text_in_training":bool(texts[text_hash]),
              "training_text_match_count":len(texts[text_hash]),
              "training_text_match_groups":len({x[1] for x in texts[text_hash]}),
              "same_text":{"codeccat":measure(label,raw,pred),"chardet":measure(label,raw,ch),"charset_normalizer":measure(label,raw,cn)},
              "competitor_predictions":{"chardet":ch,"charset_normalizer":cn}}
        cases.append(item);outgroups[text_hash].append(item["path"])
    clean=[c for c in cases if not c["identical_text_in_training"]]
    metrics={name:{"n":len(clean),"correct":sum(x["same_text"][name] is True for x in clean)}
             for name in ("codeccat","chardet","charset_normalizer")}
    result={"schema":"codeccat-semantic-transcode-audit-v1",
       "source":"uchardet fixtures; not independent source certification",
       "competitor_versions":{"chardet":chardet.__version__,"charset_normalizer":charset_normalizer.__version__},
       "all":len(cases),"training_semantic_overlaps":sum(c["identical_text_in_training"] for c in cases),
       "semantic_nonoverlap":len(clean),"metrics":metrics,
       "cross_encoding_external_groups":[v for v in outgroups.values() if len(v)>1],
       "errors":[x for x in clean if not all(y is True for y in x["same_text"].values())],
       "limits":["Text hashed from same <=4096-byte prefix; longer-document shared passages and near duplicates may remain.",
                 "No true external confirmation provenance established.","Does not promote weights."]}
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    (dest/"cases.jsonl").write_text("\n".join(json.dumps(c,sort_keys=True) for c in cases)+"\n")
    print("SEMANTIC_AUDIT_START",flush=True);print(json.dumps(result,indent=2),flush=True);print("SEMANTIC_AUDIT_END",flush=True)
if __name__=="__main__":main()
