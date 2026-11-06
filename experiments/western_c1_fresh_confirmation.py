"""Fresh-source, preregistered C1 reranking paired confirmation.

No edits to installed codeccat; only metadata-only outcomes exported.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import codecs
import hashlib
import importlib.metadata
import json
import sys

import codeccat
import polib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets" / "official"))
from audit_holdout_overlap import blocks
from overall_stratified_benchmark import WESTERN, TRAIN_HASH, digest

SOURCES = {
    "evince": ("external-evince/po", "e822030005f22231052667d51398a84327004924"),
    "epiphany": ("external-epiphany/po", "baf23d30acd0ce758632a99aa99f780f45e1802a"),
}
MANIFEST = Path("source-data/codeccat-replay-manifest.jsonl")
OUT = Path("research-results/western-c1-fresh-confirmation.json")


def guard(raw, detected):
    base = detected.text
    if base is None or not any(0x80 <= ord(c) <= 0x9f for c in base):
        return base
    for codec in detected.candidates:
        if str(codec) == str(detected.encoding):
            continue
        try:
            candidate = raw.decode(codec, "strict")
        except (UnicodeError, LookupError):
            continue
        if not any(0x80 <= ord(c) <= 0x9f for c in candidate):
            return candidate
    return base


def input_rows():
    samples = []
    excluded = defaultdict(Counter)
    for source, (directory, _) in SOURCES.items():
        files = sorted(Path(directory).glob("*.po"))
        if len(files) < 80:
            raise ValueError(f"{source} has only {len(files)} po files")
        for path in files:
            raw = path.read_bytes()
            try:
                truth = raw.decode("utf-8", "strict")
            except UnicodeError:
                excluded[source]["natural_not_utf8"] += 1
                continue
            if b"charset=utf-8" not in raw.lower():
                excluded[source]["natural_no_charset"] += 1
                continue
            samples.append((source, "natural_utf8", path.name, "utf-8", raw, truth))
        for locale, codecs in WESTERN.items():
            path = Path(directory) / (locale + ".po")
            if not path.exists():
                excluded[source]["missing_locale"] += 1
                continue
            po = polib.pofile(str(path), encoding="utf-8", check_for_duplicates=False)
            translated = "\n".join(x.msgstr for x in po if x.msgstr and not x.obsolete)
            chunks = [translated[i:i + 1024] for i in range(0, len(translated), 1024)]
            for codec in codecs:
                selected = 0
                for idx, text in enumerate(chunks):
                    if len(text) < 128 or sum(ord(c) >= 128 for c in text) < 5:
                        excluded[source]["ascii_or_short_chunk"] += 1
                        continue
                    try:
                        raw = text.encode(codec, "strict")
                    except UnicodeError:
                        excluded[source]["not_encodable"] += 1
                        continue
                    samples.append((source, "western_synthetic",
                                    f"{locale}.po#{idx}:{codec}", codec, raw, text))
                    selected += 1
                    if selected == 8:
                        break
                if not selected:
                    excluded[source]["no_selected_chunks"] += 1
    return samples, excluded


def screen_before_scoring(samples):
    if digest(MANIFEST.read_bytes()) != TRAIN_HASH:
        raise ValueError("training manifest hash mismatch")
    records = [json.loads(s) for s in MANIFEST.read_text().splitlines() if s.strip()]
    if len(records) != 3086:
        raise ValueError("historical manifest count mismatch")
    full, prefixes, spans = set(), set(), defaultdict(set)
    root = MANIFEST.parent.resolve()
    for r in records:
        rel = Path(r["path"])
        path = (root / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(root):
            raise ValueError("unsafe source path")
        data = path.read_bytes()
        if digest(data) != r["sha256"] or digest(data[:4096]) != r["prefix_sha256"]:
            raise ValueError("training bytes changed")
        full.add(r["sha256"])
        prefixes.add(r["prefix_sha256"])
        for block in blocks(data[:4096]):
            spans[block].add(r["path"])
    flagged = []
    for source, pool, path, codec, raw, truth in samples:
        nearby = Counter(p for block in blocks(raw) for p in spans.get(block, ()))
        if digest(raw) in full or digest(raw[:4096]) in prefixes or max(nearby.values(), default=0) >= 3:
            flagged.append({
                "source": source, "pool": pool, "file": path, "raw_sha256": digest(raw),
            })
    if flagged:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "status": "overlap_screen_failed_no_scoring", "flagged": flagged,
            "raw_content_exported": False,
        }, indent=2) + "\n")
        raise ValueError(f"{len(flagged)} inputs flagged for overlap; scoring aborted")


def main():
    assert importlib.metadata.version("codeccat") == "0.1.0"
    from importlib.resources import files
    assert hashlib.sha256(
        files("codeccat").joinpath("models/default.ccm").read_bytes()
    ).hexdigest() == "16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b"
    samples, excluded = input_rows()
    screen_before_scoring(samples)
    detector = codeccat.CodecCat()
    stats = defaultdict(Counter)
    codec_groups = defaultdict(Counter)
    changed = []
    for source, pool, name, codec, raw, truth in samples:
        baseline = detector.detect(raw)
        proposed_text = guard(raw, baseline)
        before = baseline.text == truth
        after = proposed_text == truth
        key = f"{source}/{pool}"
        s = stats[key]
        s["n"] += 1
        s["baseline_correct"] += int(before)
        s["counterfactual_correct"] += int(after)
        s["helped"] += int(after and not before)
        s["harmed"] += int(before and not after)
        s["baseline_silent_wrong"] += int(baseline.text is not None and not before)
        s["counterfactual_silent_wrong"] += int(proposed_text is not None and not after)
        s["changed"] += int(baseline.text != proposed_text)
        canon = codecs.lookup(codec).name
        ck = f"{source}/{canon}"
        cs = codec_groups[ck]
        cs["n"] += 1
        cs["baseline_correct"] += int(before)
        cs["counterfactual_correct"] += int(after)
        cs["helped"] += int(after and not before)
        cs["harmed"] += int(before and not after)
        if proposed_text != baseline.text:
            changed.append({
                "source": source, "pool": pool, "path": name,
                "reference_codec": canon, "original_guess": baseline.encoding,
                "sha256": digest(raw), "helped": after and not before,
                "harmed": before and not after,
            })
    report = {
        "schema": "codeccat-c1-fresh-external-pair-v1",
        "status": "screen_negative_evaluated",
        "source_commits": {k: v[1] for k, v in SOURCES.items()},
        "groups": {k: dict(v) for k,v in sorted(stats.items())},
        "by_codec": {k: dict(v) for k,v in sorted(codec_groups.items())},
        "exclusions": {k: dict(v) for k,v in sorted(excluded.items())},
        "changed_case_metadata": changed,
        "raw_content_exported": False,
        "limitations": "New source projects, but synthetic recodings of correlated translated catalogs; negative byte screen not proof of semantic independence; research-only.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("C1_FRESH_SOURCE_RESULT", json.dumps(report["groups"], sort_keys=True))


if __name__ == "__main__":
    main()
