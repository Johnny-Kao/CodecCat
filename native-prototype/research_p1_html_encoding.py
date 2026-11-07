"""P1 research only: discriminate archive-label, HTML declaration and classifier errors.

Original historical HTML bytes, frozen published Runtime 0.1.1. No model edits.
Corpus inclusion and training-overlap screen reused from preregistered C1 study.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import codecs
import hashlib
import json
import re
import sys
import zipfile

import codeccat

sys.path.insert(0, str(Path("native-prototype").resolve()))
from evaluate_c1_real_html import ARCHIVE, prepare, train_hashes, truth_from_label

OUT = Path("research-results/p1-html-encoding-diagnostic.json")
META = re.compile(
    rb"<meta\b[^>]{0,650}?\bcharset\s*=\s*['\"]?\s*([a-zA-Z0-9_\-]+)",
    flags=re.IGNORECASE,
)
ASCII_TAG = re.compile(rb"<[A-Za-z/!][\x20-\x7e]{0,450}>")
WEB_ALIASES = {
    "gb2312": "gbk", "gb_2312": "gbk", "gb_2312-80": "gbk",
    "x-gbk": "gbk", "chinese": "gbk", "cp936": "gbk",
    "x-sjis": "shift_jis", "shift-jis": "shift_jis",
    "windows-31j": "shift_jis", "ms_kanji": "shift_jis",
    "cp932": "shift_jis", "sjis": "shift_jis",
}


def canon(label: str) -> str | None:
    s = label.strip().lower().replace("_", "-")
    if s in WEB_ALIASES:
        return WEB_ALIASES[s]
    try:
        return codecs.lookup(s).name
    except LookupError:
        return None


def declared(raw: bytes) -> str | None:
    # Approximate metadata-only lexical scan, NOT a WHATWG HTML parser.
    m = META.search(raw[:1024])
    return m.group(1).decode("ascii", "ignore").lower() if m else None


def source_mask(raw: bytes) -> tuple[bytes, int]:
    # Counterfactual only: strips ASCII-only HTML tags without touching other
    # bytes, preserving byte-level character encoding outside matched tags.
    result, count = ASCII_TAG.subn(b" ", raw)
    return result, count


def main() -> None:
    full = train_hashes()
    detector = codeccat.CodecCat()
    by_group: dict[str, Counter] = defaultdict(Counter)
    miss_classes: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list] = defaultdict(list)
    with zipfile.ZipFile(ARCHIVE) as z:
        eligible, exclusions = prepare(z, full)
        assert len(eligible) == 1886, len(eligible)
        for name, group, label, source_hash, n_bytes in eligible:
            raw = z.read(name)
            assert hashlib.sha256(raw).hexdigest() == source_hash
            actual = detector.detect(raw)
            gold = truth_from_label(raw, label)
            agreed = actual.text == gold
            s = by_group[group]
            s["n"] += 1
            s["label_agree"] += int(agreed)
            s["label_disagree"] += int(not agreed)
            d = declared(raw)
            normalized_d = canon(d) if d else None
            s["has_early_meta_charset"] += int(bool(d))
            s["meta_matches_archive_label"] += int(normalized_d == canon(label) if d else False)
            s["meta_conflicts_with_archive_label"] += int(bool(d) and bool(normalized_d) and normalized_d != canon(label))
            s["meta_unrecognized"] += int(bool(d) and normalized_d is None)
            s["label_disagree_meta_matches"] += int(not agreed and bool(d) and normalized_d == canon(label))
            s["label_disagree_meta_conflicts"] += int(not agreed and bool(d) and bool(normalized_d) and normalized_d != canon(label))
            s["label_disagree_no_meta"] += int(not agreed and not d)
            candidate = {canon(x) for x in actual.candidates}
            s["label_disagree_label_is_ranked_candidate"] += int(not agreed and canon(label) in candidate)
            s["label_disagree_label_absent_from_candidates"] += int(not agreed and canon(label) not in candidate)
            s["label_disagree_model_abstained"] += int(not agreed and actual.text is None)
            if group == "Shift_JIS":
                # CP932 and strict Python shift_jis often name the same
                # Japanese legacy family but have distinct Unicode mappings.
                try:
                    cp932_text = raw.decode("cp932", "strict")
                    s["strict_cp932_can_decode"] += 1
                    s["decoded_text_exactly_cp932"] += int(actual.text == cp932_text)
                    s["shiftjis_cp932_text_differs"] += int(cp932_text != gold)
                    s["label_disagree_exactly_cp932"] += int(
                        not agreed and actual.text == cp932_text
                    )
                except UnicodeError:
                    s["strict_cp932_unavailable"] += 1
            if group == "GBK" and actual.encoding == "gb2312":
                s["predicted_gb2312"] += 1
                try:
                    gb2312_text = raw.decode("gb2312", "strict")
                    s["gb2312_strict_supported"] += 1
                    s["gb2312_vs_gbk_unicode_differs"] += int(gb2312_text != gold)
                except UnicodeError:
                    s["gb2312_strict_unavailable"] += 1
            if not agreed:
                miss_classes[group][str(actual.encoding)] += 1
            masked, tags = source_mask(raw)
            if tags >= 3 and len(masked) > 64:
                s["masked_count"] += 1
                pred = detector.detect(masked)
                try:
                    expected_masked = truth_from_label(masked, label)
                except UnicodeError:
                    s["masked_invalid_label_decode"] += 1
                    continue
                masked_ok = pred.text == expected_masked
                s["masked_label_agree"] += int(masked_ok)
                s["masked_help"] += int(not agreed and masked_ok)
                s["masked_harm"] += int(agreed and not masked_ok)
                s["masked_still_wrong"] += int(not agreed and not masked_ok)
                if (not agreed and masked_ok) or (agreed and not masked_ok):
                    if len(examples[group]) < 10:
                        examples[group].append({
                            "sha256": source_hash, "bytes": n_bytes,
                            "label": label, "baseline_guess": actual.encoding,
                            "masked_guess": pred.encoding, "tag_replacements": tags,
                            "before_label_agree": agreed, "after_label_agree": masked_ok,
                            "early_meta": d, "source_file": name,
                        })
    report = {
        "title": "P1 HTML archive/decoder hypotheses (diagnostic, not model patch)",
        "source": "IUST-HTMLCharDet pinned historical crawled corpus",
        "runtime": "codeccat==0.1.1",
        "original_screened_count": sum(x["n"] for x in by_group.values()),
        "groups": {k:dict(v) for k,v in sorted(by_group.items())},
        "predictions_on_label_disagreeing": {
            k:dict(v.most_common(12)) for k,v in sorted(miss_classes.items())
        },
        "example_metadata_only": dict(examples),
        "exclusions": {k:dict(v) for k,v in exclusions.items()},
        "limitations": [
            "Archive encoding folder is a proxy label, not audited Unicode truth.",
            "The meta charset check is ASCII regex over first 1024 bytes and is not WHATWG prescan or HTTP header evidence.",
            "Simple ASCII HTML tag masking is a counterfactual perturbation, not a supported decoder or proof of root cause.",
            "The source files are historically correlated; no release/metric improvements are authorized.",
        ],
        "raw_html_in_artifacts": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=True)+"\n",encoding="utf-8")
    print("P1_HTML_ROOT_CAUSE_AGGREGATES",json.dumps({
        "n":report["original_screened_count"],
        "groups":report["groups"],
        "wrong_predictions":report["predictions_on_label_disagreeing"]
    },sort_keys=True,ensure_ascii=True))


if __name__ == "__main__":
    main()
