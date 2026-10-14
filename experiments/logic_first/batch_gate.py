"""Single-run, high-density Logic-First correctness and cost audit.

Exhaustive two-byte space, generated valid multilingual bytes, negative mutations,
random/adversarial bytes, truncation behavior and throughput. All use one process,
one data seed, one CI setup, one JSON evidence artifact. No R21 modifications.
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path

from codeccat.logic_first import SUPPORTED, _validate, evaluate

SEED = 20261015
TEXT = {
    "utf-8": "日本語简体中文繁體中文한국어РусскийالعربيةFrançaisΕλληνικά🗾🙂 café résumé",
    "shift-jis": "日本語ひらがなカタカナ漢字学校情報変換検査文書東京都について",
    "euc-jp": "日本語ひらがなカタカナ漢字学校情報変換検査文書東京都について",
    "big5": "繁體中文編碼資料驗證臺灣企業金融管理系統識別專業文章",
    "gb18030": "简体中文编码数据验证企业管理金融证券语言模型信息系统",
}


def available_characters(codec: str) -> str:
    result=[]
    for c in TEXT[codec]:
        try: c.encode(codec, "strict")
        except UnicodeEncodeError: continue
        result.append(c)
    return "".join(result)


def execute(exhaustive: bool, valid_per_codec: int, random_cases: int) -> dict:
    rng = random.Random(SEED)
    metrics = {c: Counter() for c in SUPPORTED}
    examples=[]
    phase_times={}
    total_checked=0

    def check(data: bytes, subset=None, category=""):
        nonlocal total_checked
        for codec in (subset or SUPPORTED):
            evidence = _validate(data,codec)
            assert evidence.status in ("valid","invalid","incomplete")
            assert 0<=evidence.consumed<=len(data)
            metrics[codec][f"{category}_n"]+=1
            metrics[codec][f"{category}_{evidence.status}"]+=1
            total_checked+=1
            try:
                data.decode(codec, "strict")
            except UnicodeDecodeError:
                metrics[codec][f"{category}_strict_reject"]+=1
            else:
                metrics[codec][f"{category}_strict_accept"]+=1
                if evidence.status != "valid":
                    metrics[codec]["false_exclusions"]+=1
                    if len(examples)<20:
                        examples.append({"codec":codec,"category":category,
                                         "bytes":data[:64].hex(),"status":evidence.status})
    start=time.perf_counter()
    if exhaustive:
        # All length-two buffers, all encodings: 65,536 * five differential checks.
        for n in range(65536):
            check(bytes((n>>8,n&255)),category="all_two_byte")
    phase_times["exhaustive_two_byte_s"]=round(time.perf_counter()-start,3)

    start=time.perf_counter()
    # Textual true positives are constructed via Python's strict reference encoder.
    # Each sample has a varying length and unique composition.
    for codec in SUPPORTED:
        alphabet=available_characters(codec)
        if not alphabet:raise RuntimeError(f"No encodable characters for {codec}")
        for i in range(valid_per_codec):
            length=rng.randint(1,144)
            chars="".join(rng.choices(alphabet,k=length))
            raw=chars.encode(codec,"strict")
            check(raw,subset=(codec,),category="generated_known_valid")
            # Mutation adversaries: changing byte may preserve strict validity.
            if i%8==0:
                b=bytearray(raw)
                b[rng.randrange(len(b))]=rng.randrange(256)
                check(bytes(b),subset=(codec,),category="mutation")
            # Prefixes can be incomplete rather than invalid; no false exclusion
            # should occur whenever the strict decoder accepts the prefix.
            if i%16==0:
                for cut in (1,2,3):
                    if len(raw)>cut:
                        check(raw[:-cut],subset=(codec,),category="truncated")
    phase_times["generated_multilingual_s"]=round(time.perf_counter()-start,3)

    start=time.perf_counter()
    # Noise also covers all-ASCII, all-high-bytes, and NUL-enriched patterns.
    for i in range(random_cases):
        length=rng.randint(0,96)
        if i%4==0:
            raw=rng.randbytes(length)
        elif i%4==1:
            raw=bytes(rng.randrange(128) for _ in range(length))
        elif i%4==2:
            raw=bytes(rng.randrange(128,256) for _ in range(length))
        else:
            raw=bytes(rng.choice((0x00,0x7f,0x80,0x81,0x8e,0x8f,0xa1,0xfe,0xff)) for _ in range(length))
        check(raw,category="random_and_adversarial")
    phase_times["random_adversarial_s"]=round(time.perf_counter()-start,3)

    # Separate one-pass measuring combined validation over all five encodings;
    # do not confuse this with a production single-route selective fast path.
    perf_inputs=[]
    for _ in range(20000):
        codec=rng.choice(SUPPORTED)
        alphabet=available_characters(codec)
        perf_inputs.append("".join(rng.choices(alphabet,k=rng.randint(10,70))).encode(codec))
    start=time.perf_counter()
    nstatus=Counter()
    for raw in perf_inputs:
        for name,evidence in evaluate(raw).items():
            nstatus[f"{name}:{evidence.status}"]+=1
    duration=time.perf_counter()-start
    phase_times["all_five_throughput_s"]=round(duration,3)

    result={
        "program":"Logic-first batched correctness gate v1",
        "seed":SEED,"python":sys.version.split()[0],
        "settings":{"exhaustive":exhaustive,"valid_per_codec":valid_per_codec,
                    "random_cases":random_cases,"performance_samples":len(perf_inputs)},
        "total_structural_evaluations":total_checked+len(perf_inputs)*len(SUPPORTED),
        "false_exclusions":sum(x["false_exclusions"] for x in metrics.values()),
        "false_exclusion_examples":examples,
        "differential_by_encoding":{k:dict(sorted(v.items())) for k,v in metrics.items()},
        "phase_seconds":phase_times,
        "all_five_evaluations_per_second":round(len(perf_inputs)*len(SUPPORTED)/duration),
        "max_rss_mib_linux":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
        "notes":[
            "Structural viability is necessary but not sufficient for strict decoding.",
            "Synthetic data and Python codec agreement are logic safety tests, NOT real-world encoding accuracy.",
            "An incomplete prefix is not evidence of an impossible candidate.",
            "No model training or inference architecture changes are evaluated by this test.",
        ],
    }
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--valid-per-codec",type=int,default=50000)
    parser.add_argument("--random-cases",type=int,default=60000)
    parser.add_argument("--skip-exhaustive",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("research-results/logic-first/batch.json"))
    args=parser.parse_args()
    result=execute(not args.skip_exhaustive,args.valid_per_codec,args.random_cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("LOGIC_BATCH_RESULT_START",flush=True)
    print(json.dumps({k:v for k,v in result.items() if k!="differential_by_encoding"},indent=2),flush=True)
    print("LOGIC_BATCH_RESULT_END",flush=True)
    if result["false_exclusions"]:
        raise SystemExit(f"FAIL: {result['false_exclusions']} strict-decodable bytes excluded by structural gate")


if __name__=="__main__":
    main()
