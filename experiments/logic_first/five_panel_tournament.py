"""Five disjoint seeded mixed-source panels, same bytes to each package.

Development benchmark, NOT independent release acceptance:
- chardet/test-data was used to select CodecCat architecture
- Enca and charset-normalizer fixtures have small counts and unverified lineage
- five panels are disjoint in raw bytes, not necessarily in semantic source text
"""
from collections import Counter,defaultdict
from pathlib import Path
import codecs,hashlib,json,random,platform,time,statistics
import chardet,charset_normalizer
from charset_normalizer import from_bytes
from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from enca_external_gate import ALIASES
from mixed_random_benchmark import LABELS
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence

SEED=20261105
SOURCES=("chardet","enca","charset-normalizer")

def strict(raw,name):
    if not name:return None
    try:
        return codecs.getincrementaldecoder(name)("strict").decode(raw,final=True).lstrip("\ufeff")
    except (UnicodeError,LookupError,ValueError,TypeError):return None

def make_cases(rows):
    seen=set()
    pool=defaultdict(list)
    full_hashes={hashlib.sha256(b).hexdigest() for b,y,g,k,sha in rows}
    prefix_hashes={hashlib.sha256(b[:4096]).hexdigest() for b,y,g,k,sha in rows}
    counts=Counter()
    for raw,label,group,fold,sha in rows:
        if strict(raw,label) is None:
            counts["chardet_invalid_full_decode"]+=1
            continue
        digest=hashlib.sha256(raw).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        pool["chardet"].append(dict(raw=raw,label=label,group=group,fold=fold,sha=digest,source="chardet"))
    for path in sorted(Path("external/enca/test").iterdir()):
        if not path.is_file() or len(path.name.split("."))!=2:continue
        label=ALIASES.get(path.name.split(".")[-1])
        if not label:continue
        raw=path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        prefix=hashlib.sha256(raw[:4096]).hexdigest()
        if strict(raw,label) is None:continue
        if digest in seen or digest in full_hashes or prefix in prefix_hashes:
            counts["enca_overlap_skipped"]+=1
            continue
        seen.add(digest)
        pool["enca"].append(dict(raw=raw,label=label,group=path.name,fold=-1,sha=digest,source="enca"))
    for name,label in sorted(LABELS.items()):
        raw=(Path("external/charset_normalizer/data")/name).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        prefix=hashlib.sha256(raw[:4096]).hexdigest()
        if strict(raw,label) is None:continue
        if digest in seen or digest in full_hashes or prefix in prefix_hashes:
            counts["charset_normalizer_overlap_skipped"]+=1
            continue
        seen.add(digest)
        pool["charset-normalizer"].append(dict(raw=raw,label=label,group=name,fold=-1,sha=digest,source="charset-normalizer"))
    return pool,counts

def distribute(pool):
    # Shuffle within each source/encoding/fold stratum and round-robin into five
    # panels. A raw byte document is assigned to exactly one panel.
    buckets=defaultdict(list)
    for source,records in pool.items():
        for r in records:
            buckets[(source,r["label"],r["fold"])].append(r)
    panels=[[] for _ in range(5)]
    rng=random.Random(SEED)
    for key in sorted(buckets):
        bucket=buckets[key]
        rng.shuffle(bucket)
        offset=rng.randrange(5)
        for i,r in enumerate(bucket):
            panels[(offset+i)%5].append(r)
    for p in panels:rng.shuffle(p)
    all_hashes=[r["sha"] for p in panels for r in p]
    assert len(all_hashes)==len(set(all_hashes)), "panels must not share exact bytes"
    return panels

def main():
    rows=corpus(Path("training"))
    pool,inventory=make_cases(rows)
    panels=distribute(pool)
    folds={}
    for fold in range(3):
        training=[(b,y,g) for b,y,g,k,sha in rows if k!=fold]
        folds[fold]=(fit(residual_subset(training)[0],"binary_high"),
                     LanguageCoherence.fit(training))
    all_train=[(b,y,g) for b,y,g,k,sha in rows]
    external_model=(fit(residual_subset(all_train)[0],"binary_high"),LanguageCoherence.fit(all_train))
    report_panels=[]
    cases=[]
    overall=defaultdict(Counter)
    by_source=defaultdict(lambda:defaultdict(Counter))
    for idx,panel in enumerate(panels):
        panel_stats=defaultdict(Counter)
        source_stats=defaultdict(lambda:defaultdict(Counter))
        paired=defaultdict(Counter)
        for r in panel:
            raw,label,source=r["raw"],r["label"],r["source"]
            expected=strict(raw,label)
            assert expected is not None
            model,language=folds[r["fold"]] if source=="chardet" else external_model
            result=detect(model,raw,language_model=language)
            cd=chardet.detect(raw).get("encoding")
            cb=from_bytes(raw).best()
            cn=cb.encoding if cb else None
            preds={"codeccat":result.encoding if result.accepted else None,
                   "chardet":cd,"charset-normalizer":cn}
            correct={"codeccat":result.accepted and result.text==expected,
                     "chardet":strict(raw,cd)==expected if cd else False,
                     "charset-normalizer":strict(raw,cn)==expected if cn else False}
            for engine,ok in correct.items():
                for bucket in (panel_stats[engine],source_stats[source][engine],
                               overall[engine],by_source[source][engine]):
                    bucket["n"]+=1;bucket["correct"]+=int(ok)
            for other in ("chardet","charset-normalizer"):
                pair=paired[other]
                pair["codeccat_only"]+=int(correct["codeccat"] and not correct[other])
                pair["competitor_only"]+=int(correct[other] and not correct["codeccat"])
            cases.append({"panel":idx+1,"sha256":r["sha"],"source":source,
                          "fold":r["fold"],"group":r["group"],"truth":label,
                          "bytes":len(raw),"predicted":preds,"correct":correct})
        panel_result={"panel":idx+1,"n":len(panel),
            "engines":{k:{**dict(v),"accuracy":round(v["correct"]/v["n"],5)}
                       for k,v in panel_stats.items()},
            "by_source":{s:{k:dict(v) for k,v in m.items()}
                         for s,m in source_stats.items()},
            "paired":{k:dict(v) for k,v in paired.items()}}
        report_panels.append(panel_result)
        print("MIXED_PANEL_DONE",idx+1,len(panel),flush=True)
    report={"seed":SEED,"panel_count":5,
        "sources":{s:len(pool[s]) for s in SOURCES},
        "inventory":dict(inventory),
        "versions":{"chardet":chardet.__version__,"charset-normalizer":charset_normalizer.__version__,
                    "python":platform.python_version()},
        "overall":{k:{**dict(v),"accuracy":round(v["correct"]/v["n"],5)}
                   for k,v in overall.items()},
        "by_source":{s:{e:{**dict(v),"accuracy":round(v["correct"]/v["n"],5)}
                         for e,v in m.items()} for s,m in by_source.items()},
        "panels":report_panels,
        "limitations":["This is development/fixture performance, not an untouched confirmation set.",
                       "Five panels partition the available bytes, but share the same source corpora and trained fold models.",
                       "Small Enca and charset-normalizer fixtures cannot support five independent large source-balanced samples.",
                       "Filename encodings, semantic duplicates, and original-document provenance are not fully authenticated.",
                       "For chardet corpus rows, CodecCat models train only on the other two source-group folds.",
                       "External rows use a full chardet-trained model; exact and 4096-byte prefix overlaps were excluded.",
                       "No optimization, tuning, release promotion, or benchmarking-based parameter selection."]}
    out=Path("research-results/five-panel-tournament")
    out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases)+"\n")
    (out/"manifest.json").write_text(json.dumps([{"panel":c["panel"],"sha256":c["sha256"],"source":c["source"],
                                                    "group":c["group"],"fold":c["fold"]} for c in cases],indent=2))
    print("FIVE_PANEL_TOURNAMENT_RESULT",json.dumps({k:v for k,v in report.items() if k!="panels"}),flush=True)
    print("FIVE_PANEL_PER_RUN",json.dumps(report["panels"]),flush=True)

if __name__=="__main__":main()
