"""Five-source-stratified corpus benchmark; installed release wheel only.

Three source pools: chardet test-data, charset-normalizer data, and independent
real translation sources (Vim/Git) previously used by CodecCat researchers.
The first is in CodecCat's training provenance. Do NOT claim external accuracy.
No fixture bytes are uploaded as GitHub Actions artifacts.
"""
from __future__ import annotations

import argparse
import base64
import codecs
import hashlib
import json
import os
import platform
import random
import re
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261106
ROOT = Path("benchmark-sources")
OUT = Path("research-results/five-corpus")
N = 5
SOURCE_REVS = {
    "chardet": "b0c0d206913ba118ffea2f7d306883e7a4a76db7",
    "charset_normalizer": "7d74fe4a8341ae27a8f42c759793301dbd304e10",
    "vim": "88d96a15f99851faca0cacbbbdcede3f8c25149b",
    "git": "0ffa3e35c7c82ab8d0de226149fb03b3f7674d05",
}
ENGINES = ("codeccat", "chardet", "charset_normalizer")


def norm(enc):
    try:
        return codecs.lookup(enc).name
    except (TypeError, LookupError):
        return None


def get_text(b, encoding):
    if not encoding:
        return None
    try:
        return b.decode(encoding, errors="strict").lstrip("\ufeff")
    except (LookupError, UnicodeError):
        return None


def record(pool, source_file, raw, label, group, kind):
    label = norm(label)
    if not label or len(raw) < 24 or len(raw) > 1024 * 1024:
        return None
    if get_text(raw, label) is None:
        return None
    return dict(pool=pool, source=source_file, data=raw, encoding=label,
                group=group, kind=kind, sha256=hashlib.sha256(raw).hexdigest())


def chardet_records():
    root = ROOT / "chardet"
    sys.path.insert(0, str(root / "scripts"))
    from check_test_data import parse_dir_name
    seen = set()
    records = []
    for p in sorted(root.glob("*/*")):
        if not p.is_file():
            continue
        enc, _ = parse_dir_name(p.parent.name)
        label = norm(enc)
        if not label:
            continue
        raw = p.read_bytes()
        if not raw:
            continue
        # Preserve complete UTF characters and avoid synthetic truncated bytes.
        raw = raw[:65536]
        while raw and get_text(raw, label) is None:
            raw = raw[:-1]
        if not raw:
            continue
        group = p.stem.lower().replace("culturax_mc4_", "culturax_").replace("culturax_oscar-2301_", "culturax_")
        item = record("chardet", p.relative_to(root).as_posix(), raw, label, group, "training_overlap")
        if item and item["sha256"] not in seen:
            records.append(item)
            seen.add(item["sha256"])
    # Balanced sampling across codec labels, not first-N in directory order.
    by_label = defaultdict(list)
    for row in records:
        by_label[row["encoding"]].append(row)
    chosen = []
    for label, items in sorted(by_label.items()):
        # Stable group ordering, with multiple files for the same source grouped later.
        items.sort(key=lambda x: hashlib.sha256(f"{SEED}:{x['source']}".encode()).digest())
        chosen += items[:min(8, len(items))]
    return chosen


def charset_normalizer_records():
    root = ROOT / "charset_normalizer" / "data"
    rows = []
    for p in sorted(root.glob("sample-*.txt")):
        b = p.read_bytes()
        # Original fixture files, no synthetic alternate labels.
        # A UTF-8 strict-decodable sample may still be ASCII-ambiguous.
        item = record("charset_normalizer", p.name, b, "utf-8-sig" if b.startswith(b"\xef\xbb\xbf") else "utf-8", p.name, "external_published_fixture")
        if item:
            rows.append(item)
    return rows


def po_records():
    rows = []
    # Analyze source-controlled original PO files. Use actual declared
    # charset for reference decoding and hide charset declarations.
    for repo, folder in (("vim", ROOT / "vim" / "src" / "po"), ("git", ROOT / "git" / "po")):
        for p in sorted(folder.glob("*.po")):
            b = p.read_bytes()
            m = re.search(br"charset=([A-Za-z0-9_.+-]+)", b, re.IGNORECASE)
            if not m:
                continue
            label = norm(m.group(1).decode("ascii"))
            if not label:
                continue
            # Remove any declaration token so the library cannot win by
            # reading the answer in a file header. Keep byte length.
            b = re.sub(br"(?i)(charset=)[A-Za-z0-9_.+-]+",
                       lambda x: x.group(1) + b"?" * (len(x.group(0)) - len(x.group(1))), b)
            # For most catalog files, the first window will contain ASCII
            # headers. Take a later window on a complete decode boundary.
            # Three fixed offsets, grouped by original document.
            offsets = [0, max(0, len(b)//3), max(0, len(b)*2//3)]
            for n, off in enumerate(offsets):
                sample = b[off:off + 8192]
                # Avoid cutting multi-byte characters at start/end:
                # move start forward and end backward as necessary.
                while sample and get_text(sample, label) is None:
                    if n > 0 and len(sample) > 256 and not sample[:1].isascii():
                        sample = sample[1:]
                    else:
                        sample = sample[:-1]
                item = record("codeccat_external", f"{repo}:{p.name}:{n}", sample, label,
                              f"{repo}/{p.name}", "external_previously_inspected")
                if item:
                    rows.append(item)
    return rows


def build_fixtures():
    import importlib.metadata
    for path, sha in ((ROOT / k, value) for k, value in SOURCE_REVS.items()):
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        assert actual == sha, (path, actual, sha)
    pools = [chardet_records(), charset_normalizer_records(), po_records()]
    print("SOURCES_COLLECTED", json.dumps({x[0]["pool"]: len(x) for x in pools if x}), flush=True)
    for pool in pools:
        if len(pool) < N:
            raise ValueError(f"source pool too small: {pool[0]['pool'] if pool else 'empty'}")
    all_rows = [r for pool in pools for r in pool]
    # Global byte dedup, then group by source doc. Assign each document to
    # exactly one corpus; no source document crosses corpus boundaries.
    seen = set()
    dedup = []
    for row in all_rows:
        if row["sha256"] not in seen:
            dedup.append(row)
            seen.add(row["sha256"])
    source_groups = defaultdict(list)
    for row in dedup:
        source_groups[(row["pool"], row["group"])].append(row)
    strata = defaultdict(list)
    for (pool, group), members in source_groups.items():
        labels = Counter(x["encoding"] for x in members)
        dominant = labels.most_common(1)[0][0]
        family = "unicode" if dominant.startswith("utf") else ("east_asian" if dominant.startswith(("shift", "euc", "gb", "big5", "cp9", "iso2022")) else "legacy")
        strata[(pool, family)].append((group, members))
    corpora = [[] for _ in range(N)]
    for key, groups in sorted(strata.items()):
        groups.sort(key=lambda entry: hashlib.sha256(f"{SEED}:{key}:{entry[0]}".encode()).digest())
        for i, (_, items) in enumerate(groups):
            corpora[i % N].extend(items)
    # Strict nonoverlap and source provenance checks.
    assert len({x["sha256"] for group in corpora for x in group}) == sum(map(len, corpora))
    for group in corpora:
        assert len(group) >= 20
        assert len(set(x["pool"] for x in group)) == 3, Counter(x["pool"] for x in group)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, group in enumerate(corpora, 1):
        group.sort(key=lambda x: x["sha256"])
        payload = [{"data": base64.b64encode(x["data"]).decode(), "encoding": x["encoding"], "pool": x["pool"],
                    "source": x["source"], "sha256": x["sha256"], "group": x["group"],
                    "kind": x["kind"]} for x in group]
        (OUT / f"corpus-{i}.json").write_text(json.dumps(payload))
        manifest.append({"corpus":i, "n":len(group), "by_pool":dict(Counter(x["pool"] for x in group)),
                         "by_kind":dict(Counter(x["kind"] for x in group)),
                         "by_codec":dict(Counter(x["encoding"] for x in group)),
                         "fixture_sha256": hashlib.sha256(json.dumps([(x["sha256"],x["encoding"]) for x in group]).encode()).hexdigest()})
    (OUT / "manifest.json").write_text(json.dumps({"seed":SEED,"source_revs":SOURCE_REVS,"corpora":manifest},indent=2))
    print("FIVE_CORPORA_BUILT",json.dumps(manifest),flush=True)


def variant_worker(engine, corpus):
    import importlib.metadata
    p = OUT / f"corpus-{corpus}.json"
    rows = json.loads(p.read_text())
    samples = [(base64.b64decode(x["data"]), x["encoding"]) for x in rows]
    expected = [get_text(b, name) for b, name in samples]
    assert all(s is not None for s in expected)
    start = time.perf_counter()
    if engine == "codeccat":
        import codeccat
        detector = codeccat.CodecCat()
        def predict(data):
            r = detector.detect(data)
            return r.encoding, r.text
    elif engine == "chardet":
        import chardet
        def predict(data):
            enc = chardet.detect(data).get("encoding")
            return enc, get_text(data, enc)
    else:
        from charset_normalizer import from_bytes
        def predict(data):
            obj = from_bytes(data).best()
            enc = obj.encoding if obj else None
            return enc, get_text(data, enc)
    init_ms = (time.perf_counter()-start)*1000
    for raw, _ in samples[:10]:
        predict(raw)
    # Timing includes the same complete detect+strict Unicode decoding work
    # and excludes the expected-text ground truth calculation.
    results=[]
    cpu=[]
    wall=[]
    for raw, label in samples:
        a=time.perf_counter_ns()
        b=time.process_time_ns()
        enc,txt=predict(raw)
        cpu.append((time.process_time_ns()-b)/1e6)
        wall.append((time.perf_counter_ns()-a)/1e6)
        results.append((norm(enc),txt))
    # Two more passes measure CPU latency separately from one-pass correctness.
    for repeat in range(2):
        for raw,_ in reversed(samples) if repeat % 2 else samples:
            predict(raw)
    count_correct=sum(txt==ref for (_,txt),ref in zip(results,expected))
    label_correct=sum(enc==norm(name) for (enc,_),(_,name) in zip(results,samples))
    null_count=sum(txt is None for _,txt in results)
    per_pool={}
    for pool in ("chardet","charset_normalizer","codeccat_external"):
        ix=[i for i,row in enumerate(rows) if row["pool"]==pool]
        if not ix:
            continue
        per_pool[pool]={"n":len(ix),
                        "correct_text":sum(results[i][1]==expected[i] for i in ix),
                        "correct_label":sum(results[i][0]==norm(samples[i][1]) for i in ix),
                        "cpu_ms":round(sum(cpu[i] for i in ix),3)}
    output={
        "engine":engine,"corpus":corpus, "n":len(samples), "per_pool":per_pool,
        "correct_text":count_correct, "correct_label":label_correct,
        "abstained":null_count, "accuracy_text_pct":round(count_correct/len(samples)*100,3),
        "accuracy_label_pct":round(label_correct/len(samples)*100,3),
        "wall_p50_ms":round(statistics.median(wall),4),
        "wall_p95_ms":round(sorted(wall)[int((len(wall)-1)*0.95)],4),
        "cpu_total_ms":round(sum(cpu),3),
        "cpu_p50_ms":round(statistics.median(cpu),4),
        "peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,3),
        "init_ms":round(init_ms,3),"python":platform.python_version(),
        "package_version":importlib.metadata.version("codeccat" if engine=="codeccat" else "chardet" if engine=="chardet" else "charset-normalizer"),
    }
    (OUT / f"{corpus}-{engine}.json").write_text(json.dumps(output,indent=2)+"\n")
    print("FIVE_CORPUS_RESULT",json.dumps(output),flush=True)


def summary():
    manifest=json.loads((OUT/"manifest.json").read_text())
    matrix={}
    for n in range(1,6):
        matrix[str(n)]={e:json.loads((OUT/f"{n}-{e}.json").read_text()) for e in ENGINES}
    totals={}
    for e in ENGINES:
        docs=[matrix[str(n)][e] for n in range(1,6)]
        cnt=sum(x["n"] for x in docs)
        totals[e]={"n":cnt,"correct_text":sum(x["correct_text"] for x in docs),
                   "correct_label":sum(x["correct_label"] for x in docs),
                   "accuracy_text_pct":round(sum(x["correct_text"] for x in docs)/cnt*100,3),
                   "accuracy_label_pct":round(sum(x["correct_label"] for x in docs)/cnt*100,3),
                   "cpu_total_ms":round(sum(x["cpu_total_ms"] for x in docs),3),
                   "wall_p50_by_corpus_ms":[x["wall_p50_ms"] for x in docs],
                   "rss_mib_by_corpus":[x["peak_rss_mib"] for x in docs]}
    pooled={}
    for pool in ("chardet","charset_normalizer","codeccat_external"):
        pooled[pool]={}
        for e in ENGINES:
            parts=[matrix[str(n)][e]["per_pool"].get(pool,{"n":0,"correct_text":0,"correct_label":0,"cpu_ms":0}) for n in range(1,6)]
            count=sum(p["n"] for p in parts)
            pooled[pool][e]={"n":count,"correct_text":sum(p["correct_text"] for p in parts),
                             "correct_label":sum(p["correct_label"] for p in parts),
                             "text_pct":round(100*sum(p["correct_text"] for p in parts)/count,3) if count else None,
                             "cpu_total_ms":round(sum(p["cpu_ms"] for p in parts),3)}
    result={"schema":"codeccat-five-corpus-v1","manifest":manifest,"per_corpus":matrix,"by_source_pool":pooled,
            "overall":totals,
            "limitations":["chardet/test-data is CodecCat training data: accuracy is contaminated for that pool",
                           "Vim/Git sources were previously investigated, not fresh unseen blind data",
                           "charset-normalizer text fixture labels assume strict UTF-8 (or UTF-8-SIG for BOM)",
                           "Source documents grouped but near-duplicate content between upstream repositories not exhaustively checked",
                           "Each detector runs separately on same Linux GitHub-hosted runner hardware; memory is process-wide RSS",
                           "Correct decoded Unicode text, not historical codec-name match, is primary metric",
                           "This is an internal comparative benchmark, not a statistically representative real-world accuracy estimate"]}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("FIVE_CORPUS_SUMMARY",json.dumps({"manifest":manifest["corpora"],"overall":totals,"by_source_pool":pooled}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prepare",action="store_true")
    p.add_argument("--worker",choices=ENGINES)
    p.add_argument("--corpus",type=int,choices=range(1,6))
    p.add_argument("--summary",action="store_true")
    a=p.parse_args()
    if a.prepare: build_fixtures()
    elif a.worker and a.corpus: variant_worker(a.worker,a.corpus)
    elif a.summary: summary()
    else:p.error("choose --prepare, --worker/--corpus, or --summary")
