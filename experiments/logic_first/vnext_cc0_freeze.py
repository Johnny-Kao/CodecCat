"""Freeze a new, independent, CC0 sentence source BEFORE examining model scores.

No detection/ML dependencies. Source archives and stable selected sentence IDs
are printed, hashed and archived; later test runs MUST enforce the SHA256.
This is a synthetic re-encoding generalization check, not natural prevalence.
"""
from __future__ import annotations
import argparse
import bz2
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = {
    "eng": ("cp037", "cp500", "cp1252", "iso8859-1", "utf-8"),
    "fra": ("cp1252", "iso8859-1", "iso8859-15", "iso8859-16", "utf-8"),
    "rus": ("cp1251", "cp866", "koi8-r", "utf-8"),
}
MAX_PER_CODEC = 40
SEED = "codeccat-fresh-cc0-source-20261101-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get_archive(lang):
    url = f"https://downloads.tatoeba.org/exports/per_language/{lang}/{lang}_sentences_CC0.tsv.bz2"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"CodecCat Research/1.0"}), timeout=60) as response:
        blob = response.read(5_000_001)
    if len(blob) > 5_000_000:
        raise RuntimeError("CC0 archive unexpectedly large")
    if not blob.startswith(b"BZh"):
        raise ValueError("download was not bzip2")
    return url, blob


def select(blob, lang):
    text = bz2.decompress(blob).decode("utf-8", "strict")
    codecs = SOURCES[lang]
    pool = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        sid, sentence = fields[0], fields[1]
        if not sid.isdecimal() or not (48 <= len(sentence) <= 350):
            continue
        if "\x00" in sentence or "\ufffd" in sentence:
            continue
        ident = sha((SEED + "|" + lang + "|" + sid + "|" + sentence).encode("utf-8"))
        pool.append((ident, sid, sentence))
    selected, used_text = [], set()
    quota = {name: 0 for name in codecs}
    for digest, sid, sentence in sorted(pool):
        tid = sha(sentence.encode("utf-8"))
        if tid in used_text:
            continue
        # Deterministic, precommitted target: one historical encoding per
        # sentence. Never duplicate a sentence across encoding labels.
        candidates = [codecs[(int(digest[:8], 16) + i) % len(codecs)] for i in range(len(codecs))]
        for label in candidates:
            if quota[label] >= MAX_PER_CODEC:
                continue
            try:
                raw = sentence.encode(label, "strict")
                if raw.decode(label, "strict") != sentence:
                    continue
            except (UnicodeError, LookupError):
                continue
            selected.append({"lang":lang,"sentence_id":sid,"text":sentence,
                             "codec":label,"raw_sha256":sha(raw),"text_sha256":tid})
            quota[label] += 1
            used_text.add(tid)
            break
    return selected, quota, len(pool)


def run(args):
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    all_rows, manifests = [], []
    for lang in SOURCES:
        url, blob = get_archive(lang)
        (output / f"{lang}_sentences_CC0.tsv.bz2").write_bytes(blob)
        rows, quota, count = select(blob, lang)
        if len(rows) < 60:
            raise RuntimeError(f"insufficient CC0 examples for {lang}: {len(rows)}")
        all_rows.extend(rows)
        manifests.append({
            "lang":lang,"source_url":url,"archive_sha256":sha(blob),
            "archive_bytes":len(blob),"eligible_sentences":count,
            "selected":len(rows),"selected_codec_counts":quota
        })
    report = {
        "schema": "codeccat-tatoeba-cc0-frozen-v1",
        "pre_registration": True,
        "selection_seed": SEED,
        "max_per_codec": MAX_PER_CODEC,
        "data_license": "CC0 subset only; see Tatoeba per-language _sentences_CC0 export",
        "original_source": "independent user-authored Tatoeba CC0 sentences",
        "generated_label_origin": "strictly re-encoding author text into a selected codec",
        "source_warnings": [
            "Synthetic reencoding is not naturally observed codec prevalence",
            "Sentences from the same source may be correlated",
            "Limited to 3 languages and historical codec families",
            "A later test must reject any exact decoded-text overlap with training",
        ],
        "sources": manifests,
        "n":len(all_rows),
        "selected_digest": sha(json.dumps(all_rows, ensure_ascii=False,sort_keys=True).encode("utf-8")),
    }
    (output / "frozen-source-manifest.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (output / "selected.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False,sort_keys=True) for x in all_rows)+"\n",encoding="utf-8")
    print("CC0_FROZEN_SOURCE_START", flush=True)
    print(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2),flush=True)
    print("CC0_FROZEN_SOURCE_END", flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="research-results/cc0-frozen-source")
    run(p.parse_args())
