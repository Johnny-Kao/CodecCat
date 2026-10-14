"""H12 Japanese encodings: byte validity/Unicode-script discriminability diagnostic.
Frozen R21 only, no promotion. Collect bytes anew; results do not define a new holdout.
"""
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
from codeccat import load_bundle
from codeccat.runtime import Runtime,_signals
from h2_fused_candidate_mask_ab import collect,normalize

CRAWLS=("CC-MAIN-2024-22","CC-MAIN-2024-46","CC-MAIN-2024-51")
SHA="e792df1be4a4a07f052c7786e61122d547a3c87501ad5b40f729621e8019fb54"
ENCODINGS={"shift-jis":"shift_jis","cp932":"cp932","euc-jp":"euc_jp","utf-8":"utf-8","gb18030":"gb18030","big5":"big5","cp1251":"cp1251"}
def script_score(text):
    n=max(1,len(text))
    hira=sum('\u3040'<=c<='\u309f' for c in text)
    kata=sum('\u30a0'<=c<='\u30ff' for c in text)
    kanji=sum('\u4e00'<=c<='\u9fff' for c in text)
    cyr=sum('\u0400'<=c<='\u04ff' for c in text)
    bad=sum(ord(c)<32 and c not in '\n\r\t' for c in text)
    return {"hira":hira,"kata":kata,"kanji":kanji,"cyrillic":cyr,"control":bad,
            "japanese_markers_per_1000":round(1000*(hira+kata)/n,2)}
def decode_features(data):
    out={}
    # Keep a fixed bounded prefix. Drop up to four terminal bytes to account for cut multibyte units.
    head=data[:4096]
    for name,codec in ENCODINGS.items():
        decoded=None
        cuts=None
        for tail in range(0,min(4,len(head))+1):
            try:
                decoded=head[:len(head)-tail].decode(codec,"strict")
                cuts=tail
                break
            except UnicodeDecodeError:
                pass
        out[name]={"strict_possible":decoded is not None,"trim_bytes":cuts}
        if decoded is not None:out[name].update(script_score(decoded))
    return out

def aggregate(records):
    n=len(records)
    result={"n":n,"codec_success":{},"codec_script_support":{}}
    for name in ENCODINGS:
        possible=[r["decode"][name] for r in records if r["decode"][name]["strict_possible"]]
        result["codec_success"][name]=len(possible)
        result["codec_script_support"][name]=sum(x["hira"]+x["kata"]>=2 for x in possible)
    return result

def main(a):
    assert hashlib.sha256(Path(a.model).read_bytes()).hexdigest()==SHA
    rt=Runtime(load_bundle(a.model))
    seen=set();rows=[]
    for crawl in CRAWLS:
        samples,_,_=collect(crawl)
        for row in samples:
            data=row["data"]
            digest=hashlib.sha256(data).hexdigest()
            if digest in seen:continue
            seen.add(digest)
            truth=normalize(row["label"])
            if _signals(data).route!="RH":continue
            pred=rt.rank(data)[0]
            rows.append({"hash":digest,"crawl":crawl,"truth":truth,"pred":pred,
                         "byte_len":len(data),"decode":decode_features(data)})
        print("H12_CRAWL",crawl,len(samples),flush=True)
    japanese=[r for r in rows if r["truth"] in ("shift-jis","euc-jp")]
    other=[r for r in rows if r["truth"] not in ("shift-jis","euc-jp")]
    wrong=[r for r in japanese if r["pred"]!=r["truth"]]
    label_possible=sum(r["decode"][r["truth"]]["strict_possible"] for r in japanese)
    label_support=sum(r["decode"][r["truth"]].get("hira",0)+r["decode"][r["truth"]].get("kata",0)>=2 for r in japanese)
    result={"crawls":CRAWLS,"limitations":"Research-exposed 2024 crawls; labels from collector declarations, not independently verified",
            "all_rh":aggregate(rows),"japanese_truth":aggregate(japanese),
            "japanese_wrong":aggregate(wrong),"other_truth":aggregate(other),
            "japanese_n":len(japanese),"japanese_wrong_n":len(wrong),
            "declared_japanese_strict_possible":label_possible,
            "declared_japanese_script_supported":label_support,
            "wrong_by_label":dict(Counter(r["truth"] for r in wrong)),
            "wrong_pred":dict(Counter(r["pred"] for r in wrong)),
            "wrong_cases":[r for r in wrong][:35]}
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/"diagnostic.json").write_text(json.dumps(result,indent=2))
    printable={k:v for k,v in result.items() if k!="wrong_cases"}
    print("H12_RESULT_START")
    print(json.dumps(printable,indent=2))
    print("H12_RESULT_END")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--out",default="research-results/h12")
    main(p.parse_args())
