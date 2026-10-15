"""Gate 3/5 — single runner, wide diagnostic on known-encoding bytes.

Fixture corpus is chardet's OWN public test set; benchmark is exploratory and
favors chardet. No tuning, no training, no use of results as untouched holdout.
"""
from __future__ import annotations
import argparse,codecs,hashlib,json,random,re,resource,statistics,subprocess,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import chardet,charset_normalizer
from charset_normalizer import from_bytes
from codeccat import load_bundle
from codeccat.runtime import Runtime
from h2_fused_candidate_mask_ab import normalize
from h14_japanese_fastpath import fast as h14
from gate1_architecture_tournament import late,jp_contrast

SEED=20261015
MODEL_SHA="e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54"
TESTDATA_SHA="b0c0d206913ba118ffea2f7d306883e7a4a76db7"
RE_CHARSET=re.compile(br'(?i)(?:charset|encoding)\s*=\s*["\x27]?([a-zA-Z0-9_.+-]+)')
POLICIES=("r21","h14_strict","japanese_contrast","meta_guard","metadata_only",
          "chardet_7_6","charset_normalizer_3_5")
SIZE_CAP=65536

def metadata(raw):
    seen=[normalize(m.group(1).decode("ascii","ignore")) for m in RE_CHARSET.finditer(raw[:4096])]
    return next(iter(set(seen))) if len(set(seen))==1 and seen else None

def without_metadata(raw):
    # Replace only the encoding declaration token, preserve byte length.
    head=raw[:4096]
    edits=list(RE_CHARSET.finditer(head))
    if not edits:return raw
    b=bytearray(raw)
    for m in edits:
        lo,hi=m.span(1)
        b[lo:hi]=b"?"*(hi-lo)
    return bytes(b)

def label_equivalent(actual,pred,raw):
    if not pred:return False
    try:return raw.decode(actual,"strict")==raw.decode(pred,"strict")
    except (UnicodeDecodeError,LookupError):return False

def strict_full(raw,enc):
    try:raw.decode(enc,"strict");return True
    except (UnicodeDecodeError,LookupError):return False

def predict(rt,data,policy,known=None):
    if policy=="chardet_7_6":
        out=chardet.detect(data).get("encoding")
        return normalize(out) if out else None
    if policy=="charset_normalizer_3_5":
        best=from_bytes(data).best()
        return normalize(best.encoding) if best else None
    rank=rt.rank(data);base=rank[0]
    if policy=="r21":return base
    if policy=="h14_strict":
        h=h14(data,base)[0]
        return late(data,(h,)+tuple(c for c in rank if c!=h),"strict")
    if policy=="japanese_contrast":return jp_contrast(data,base)
    declared=metadata(data)
    if policy=="metadata_only":
        return declared if declared and strict_full(data,declared) else None
    if policy=="meta_guard":
        # Conservative metadata: no override if not strict-decodable;
        # otherwise trust unique syntactic declaration, BOM is left to R21.
        if declared and strict_full(data,declared):return declared
        return predict(rt,data,"h14_strict")
    raise KeyError(policy)

def sample_directory(root):
    # Import upstream's own 2026 parsing rules for locale suffixes.
    sys.path.insert(0,str(root/"scripts"))
    from check_test_data import parse_dir_name,resolve_encoding
    records=[];dupes=set();excluded=Counter()
    for f in sorted(root.glob("*/*")):
        if not f.is_file():continue
        enc,language=parse_dir_name(f.parent.name)
        resolved=resolve_encoding(enc)
        if resolved is None:
            excluded["unsupported_codec_name"]+=1;continue
        raw=f.read_bytes()
        if not raw:
            excluded["empty"]+=1;continue
        digest=hashlib.sha256(raw).hexdigest()
        if digest in dupes:
            excluded["same_byte_duplicate"]+=1;continue
        dupes.add(digest)
        try:
            decoded=raw.decode(resolved,"strict")
        except (UnicodeDecodeError,LookupError):
            excluded["label_fails_strict_full_decode"]+=1
            continue
        try:roundtrip=decoded.encode(resolved)==raw
        except (UnicodeEncodeError,LookupError):roundtrip=False
        source="transcoded" if f.name.startswith("culturax_") else (
            "suite" if any(x in f.name for x in ("_ude_","_enca_","_uchardet_",
                                                   "_mozilla_","_chromium_")) else "mixed_wild_or_other")
        sample=raw[:SIZE_CAP]
        records.append({"name":f.relative_to(root).as_posix(),"label":normalize(enc),
            "real_codec":resolved,"language":language or "unspecified",
            "source":source,"full_bytes":len(raw),"truncated":len(raw)>SIZE_CAP,
            "roundtrip":roundtrip,"ascii_only":all(b<128 for b in sample),
            "sha":digest,"_data":sample})
    return records,excluded

def compute_stats(records,mode,only_supported=False,supported=None):
    sub=[r for r in records if mode in r["views"] and
         (not only_supported or r["label"] in supported)]
    n=len(sub);res={}
    for policy in POLICIES:
        preds=[r["views"][mode][policy] for r in sub]
        hit=sum(p==r["label"] for p,r in zip(preds,sub))
        equiv=sum(label_equivalent(r["real_codec"],p,r["_data"] if mode=="raw" else without_metadata(r["_data"])) for p,r in zip(preds,sub))
        res[policy]={"n":n,"correct":hit,"top1_exact":round(hit/max(n,1),5),
                    "decoded_text_equiv_n":equiv}
    return res

def summarize_subgroup(rows,key,supported):
    result={}
    for value in sorted(set(r[key] for r in rows),key=str):
        sub=[r for r in rows if r[key]==value]
        result[str(value)]={"n":len(sub),
              "supported_n":sum(r["label"] in supported for r in sub),
              "exact":{p:sum(r["views"]["raw"][p]==r["label"] for r in sub) for p in POLICIES}}
    return result

def perform(a):
    model=Path(a.model)
    if hashlib.sha256(model.read_bytes()).hexdigest()!=MODEL_SHA:raise RuntimeError("R21 digest mismatch")
    root=Path(a.fixtures)
    sha=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if sha!=TESTDATA_SHA:raise RuntimeError(f"Fixture revision moved: {sha}")
    rt=Runtime(load_bundle(model))
    supported=set(rt.bundle.classes)
    docs,excluded=sample_directory(root)
    if len(docs)<1000:raise RuntimeError(f"Too few fixtures: {len(docs)}")
    print("GATE3_CORPUS",len(docs),dict(excluded),flush=True)
    for idx,r in enumerate(docs):
        data=r["_data"]
        scrub=without_metadata(data)
        r["metadata"]=metadata(data)
        r["has_bom"]=data.startswith((b"\xef\xbb\xbf",b"\xff\xfe",b"\xfe\xff",b"\x00\x00\xfe\xff",b"\xff\xfe\x00\x00"))
        r["views"]={}
        # Full same-byte raw vs metadata scrub comparison.
        for view,raw in (("raw",data),("scrubbed",scrub)):
            r["views"][view]={p:predict(rt,raw,p) for p in POLICIES}
        # Prefix behavior on a stable 1/6 sample, plus all Japanese/ISO cases.
        if idx%6==0 or r["label"] in ("shift-jis","euc-jp","iso-8859-1","iso-8859-2"):
            for sz in (128,1024,4096):
                r["views"]["prefix_"+str(sz)]={p:predict(rt,data[:sz],p) for p in POLICIES}
    print("GATE3_ALL_VIEWS_FINISHED",len(docs),flush=True)
    rng=random.Random(SEED);perf_rows=rng.sample(docs,min(256,len(docs)))
    perf={}
    for policy in POLICIES:
        ns=[]
        for j in range(3):
            order=perf_rows if j%2==0 else list(reversed(perf_rows))
            start=time.perf_counter_ns()
            for r in order:predict(rt,r["_data"],policy)
            ns.append((time.perf_counter_ns()-start)/len(order))
        perf[policy]={"median_ns_per_sample":round(statistics.median(ns)),
                      "repetitions":3,"samples_per_repeat":len(perf_rows)}
    report={"gate":3,"source":"chardet/test-data","git_sha":sha,
       "explicit_warning":"This is chardet's own test corpus; strongly competitor-favorable and NOT independent evidence for product release.",
       "label_rule":"directory labels strict Python-decoded in entirety; non-decodable excluded",
       "exclusions":dict(excluded),"N":len(docs),
       "supported_model_labels":sorted(supported),
       "coverage":{"supported_n":sum(r["label"] in supported for r in docs),
                   "unsupported_n":sum(r["label"] not in supported for r in docs),
                   "ascii_ambiguous_n":sum(r["ascii_only"] for r in docs),
                   "roundtrip_n":sum(r["roundtrip"] for r in docs),
                   "declaration_n":sum(bool(r["metadata"]) for r in docs),
                   "bom_n":sum(r["has_bom"] for r in docs)},
       "versions":{"chardet":chardet.__version__,
                   "charset_normalizer":charset_normalizer.__version__},
       "raw_all":compute_stats(docs,"raw"),
       "raw_model_supported":compute_stats(docs,"raw",True,supported),
       "scrubbed_all":compute_stats(docs,"scrubbed"),
       "prefix":{
           str(sz):compute_stats(docs,"prefix_"+str(sz),True,supported) for sz in (128,1024,4096)},
       "breakdown":{
           k:summarize_subgroup(docs,k,supported)
           for k in ("label","language","source","ascii_only","roundtrip","metadata","has_bom")},
       "performance":perf,
       "mixed_process_maxrss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
       "limitations":"narrow normalized exact labels may be unidentifiable from bytes; decoded-string equivalence secondary. Some test-data fixtures share source texts across encodings. Metadata scrub is a transformation, not natural no-declaration data."}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"trace.jsonl").write_text("\n".join(json.dumps({k:v for k,v in r.items() if k!="_data"}) for r in docs)+"\n")
    print("GATE3_RESULT_START",flush=True)
    print(json.dumps({k:report[k] for k in ("N","coverage","raw_all","raw_model_supported","scrubbed_all","prefix","performance")},indent=2),flush=True)
    print("GATE3_RESULT_END",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True);p.add_argument("--fixtures",required=True)
    p.add_argument("--out",default="research-results/vnext-gate3")
    a=p.parse_args()
    perform(a)
