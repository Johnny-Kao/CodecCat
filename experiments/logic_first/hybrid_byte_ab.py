"""One preregistered hybrid-byte representation A/B for CodecCat research.

Keep TF-IDF byte-bigram baseline and augment it with a small deterministic
byte histogram, position-bucket and structural feature vector. Identical L1,
L3, grouped folds and five fixed panels. No production changes.
"""
from __future__ import annotations
import json,time,statistics,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from five_panel_tournament import make_cases,distribute,strict,SEED
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence

SCALE=0.5
# 32 byte bins, 16 high-bit occupancy by position, 8 structural summaries.
N_FEATURES=56

def side_features(s):
    raw=s.encode("latin1")
    b=raw[:4096]
    n=len(b)
    out=np.zeros(N_FEATURES,dtype=np.float64)
    if not n:return out
    data=np.frombuffer(b,dtype=np.uint8)
    hist=np.bincount(data//8,minlength=32).astype(np.float64)
    out[:32]=hist/n
    for i in range(16):
        partition=data[i::16]
        if len(partition):out[32+i]=np.count_nonzero(partition>=128)/len(partition)
    out[48]=np.count_nonzero(data==0)/n
    out[49]=np.count_nonzero(data<32)/n
    out[50]=np.count_nonzero((data>=128)&(data<=159))/n
    out[51]=np.count_nonzero(data>=128)/n
    out[52]=np.count_nonzero((data>=32)&(data<127))/n
    out[53]=len(np.unique(data))/256
    probabilities=hist[hist>0]/n
    out[54]=-np.sum(probabilities*np.log2(probabilities))/5.0
    out[55]=math.log1p(n)/math.log1p(4096)
    return out*SCALE

class HybridVectorizer:
    def __init__(self,base):
        self.base=base
    def transform(self,strings):
        x=self.base.transform(strings)
        extra=csr_matrix(np.vstack([side_features(s) for s in strings]))
        return hstack((x,extra),format="csr")

def fit_hybrid(training):
    strings=[b[:4096].decode("latin1") for b,y,g in training]
    labels=[y for b,y,g in training]
    vec=TfidfVectorizer(analyzer="char",ngram_range=(1,2),min_df=2,
            max_features=60000,sublinear_tf=True,binary=True)
    original=vec.fit_transform(strings)
    extras=csr_matrix(np.vstack([side_features(s) for s in strings]))
    x=hstack((original,extras),format="csr")
    clf=SGDClassifier(loss="modified_huber",alpha=.0003,random_state=20261015,max_iter=1500)
    clf.fit(x,labels)
    return TrainableDetector(vectorizer=HybridVectorizer(vec),classifier=clf,
                             fingerprint="hybrid-byte-statistical-research")

def add(c,baseline,hybrid):
    c["n"]+=1;c["baseline"]+=int(baseline);c["hybrid"]+=int(hybrid)
    c["fixed"]+=int(hybrid and not baseline)
    c["broken"]+=int(baseline and not hybrid)

def summary(c):
    return {**dict(c),"baseline_pct":round(c["baseline"]/c["n"]*100,3),
        "hybrid_pct":round(c["hybrid"]/c["n"]*100,3)}

def main():
    rows=corpus(Path("training"))
    pool,_=make_cases(rows);panels=distribute(pool)
    trained={}
    for fold in (0,1,2,-1):
        train=[(b,y,g) for b,y,g,k,_ in rows if k!=fold] if fold!=-1 else [(b,y,g) for b,y,g,k,_ in rows]
        residual=residual_subset(train)[0]
        trained[fold]=(fit(residual,"binary_high"),fit_hybrid(residual),LanguageCoherence.fit(train))
        print("HYBRID_FOLD_TRAINED",fold,flush=True)
    overall=Counter();by_source=defaultdict(Counter);by_length=defaultdict(Counter)
    rounds=[];cases=[];times={"base":[],"hybrid":[]}
    for idx,panel in enumerate(panels,1):
        pc=Counter()
        for row in panel:
            b=row["raw"];truth=strict(b,row["label"])
            normal,hybrid,lang=trained[row["fold"]]
            t=time.process_time_ns()
            a=detect(normal,b,language_model=lang)
            t1=time.process_time_ns()
            z=detect(hybrid,b,language_model=lang)
            t2=time.process_time_ns()
            times["base"].append((t1-t)/1e6);times["hybrid"].append((t2-t1)/1e6)
            a_ok=bool(a.accepted and a.text==truth)
            z_ok=bool(z.accepted and z.text==truth)
            for group in (overall,by_source[row["source"]],
                          by_length["short<=256" if len(b)<=256 else "long>256"],pc):
                add(group,a_ok,z_ok)
            if a.encoding!=z.encoding or a.accepted!=z.accepted:
                cases.append({"sha256":row["sha"],"panel":idx,"source":row["source"],
                              "truth":row["label"],"length":len(b),
                              "baseline":a.encoding,"hybrid":z.encoding,
                              "fixed":z_ok and not a_ok,"broken":a_ok and not z_ok})
        rounds.append({"panel":idx,**summary(pc)})
        print("HYBRID_PANEL_DONE",idx,flush=True)
    report={"seed":SEED,"features":"32 8-byte histogram bins + 16 high-bit positional buckets + 8 summaries",
            "scale":SCALE,"total":summary(overall),
            "by_source":{k:summary(v) for k,v in by_source.items()},
            "by_length":{k:summary(v) for k,v in by_length.items()},
            "panels":rounds,
            "cpu_ms_median":{k:round(statistics.median(v),4) for k,v in times.items()},
            "cpu_ms_p95":{k:round(sorted(v)[int((len(v)-1)*.95)],4) for k,v in times.items()},
            "changed":len(cases),
            "limitations":["development fixture families, not an independent blind confirmation",
                           "CPU measurements are process CPU in same worker with trained models loaded",
                           "both variants use same architecture L1 and L3; only L2 representation differs"]}
    out=Path("research-results/hybrid-byte-ab");out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"changed.jsonl").write_text("\n".join(json.dumps(x) for x in cases)+"\n")
    print("HYBRID_BYTE_AB_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
