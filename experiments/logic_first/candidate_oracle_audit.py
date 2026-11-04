"""Research-only oracle: identify absent candidates versus incorrect final selection.

No predictor training changes, no oracle decisions in runtime. Canonical correctness
is decoded-text equality; labels alone are not an adequate oracle when codecs alias.
"""
from collections import Counter
from pathlib import Path
import codecs
import hashlib
import json

from architecture_attribution import fit
from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from layered_architecture_gate import outside
from l2_variant_matrix import valid
from codeccat.vnext_byte_families import ByteFamilies, candidate_union
from codeccat.vnext_layered_pipeline import detect
from codeccat.vnext_l3_language import LanguageCoherence


def strict_text(raw, name):
    try:
        return codecs.getincrementaldecoder(name)("strict").decode(raw, final=True).lstrip("\ufeff")
    except (UnicodeError, LookupError, ValueError, TypeError):
        return None


def audit(source, name, raw, truth, model, language, families, counts, cases):
    expected = valid(raw, truth)
    if expected is None:
        return
    ranked = model.rank(raw, final=len(raw) < 4096)
    layered = detect(model, raw, language_model=language)
    candidate_sets = {
        "top3": tuple(ranked.candidates),
        "top3_plus8": candidate_union(raw, ranked, families, limit=8),
        "top3_plus16": candidate_union(raw, ranked, families, limit=16),
    }
    # Runtime may take a document declaration / L3 decision outside primary top3.
    if layered.encoding:
        candidate_sets["runtime_plus16"] = tuple(dict.fromkeys(
            (*candidate_sets["top3_plus16"], layered.encoding)
        ))
    else:
        candidate_sets["runtime_plus16"] = candidate_sets["top3_plus16"]

    base_ok = valid(raw, ranked.encoding) == expected
    runtime_ok = layered.accepted and layered.text == expected
    reached = {key: any(valid(raw, enc) == expected for enc in names)
               for key, names in candidate_sets.items()}
    c = counts[source]
    c["n"] += 1
    c["base_correct"] += int(base_ok)
    c["runtime_correct"] += int(runtime_ok)
    c["both_wrong"] += int(not base_ok and not runtime_ok)
    c["runtime_fixed"] += int(not base_ok and runtime_ok)
    c["runtime_broken"] += int(base_ok and not runtime_ok)
    for key, value in reached.items():
        c[key + "_reachable"] += int(value)
        c[key + "_missing"] += int(not value)
        c[key + "_reachable_but_runtime_wrong"] += int(value and not runtime_ok)
    c["newly_reachable_16_vs_top3"] += int(reached["top3_plus16"] and not reached["top3"])
    strict = strict_text(raw, truth)
    if strict is not None:
        c["strict_valid"] += 1
        c["strict_runtime_correct"] += int(layered.accepted and layered.text == strict)
    if not runtime_ok:
        category = ("missing_from_union16" if not reached["runtime_plus16"]
                    else "selection_error")
        c[category] += 1
        cases.append({
            "source": source, "name": name, "sha256_prefix": hashlib.sha256(raw).hexdigest()[:16],
            "truth_codec": truth, "baseline": ranked.encoding,
            "runtime": layered.encoding, "runtime_reason": layered.reason,
            "category": category, "truth_reachable_top3": reached["top3"],
            "truth_reachable_union16": reached["top3_plus16"],
            "truth_reachable_runtime_union16": reached["runtime_plus16"],
            "candidates16": list(candidate_sets["top3_plus16"]),
        })


def main():
    rows = corpus(Path("training"))
    counts = __import__("collections").defaultdict(Counter)
    cases = []
    for fold in range(3):
        train = [(b, y, g) for b, y, g, k, sha in rows if k != fold]
        model = fit(residual_subset(train)[0], "binary_high")
        language = LanguageCoherence.fit(train)
        families = ByteFamilies.fit(train)
        for raw, label, group, k, sha in rows:
            if k == fold:
                audit("development", sha, raw, label, model, language, families, counts, cases)

    train = [(b, y, g) for b, y, g, k, sha in rows]
    model = fit(residual_subset(train)[0], "binary_high")
    language = LanguageCoherence.fit(train)
    families = ByteFamilies.fit(train)
    seen = {hashlib.sha256(b).hexdigest() for b, y, g, k, sha in rows}
    prefixes = {hashlib.sha256(b[:4096]).hexdigest() for b, y, g, k, sha in rows}
    for source, name, raw, label in outside(rows):
        if hashlib.sha256(raw).hexdigest() in seen or hashlib.sha256(raw[:4096]).hexdigest() in prefixes:
            continue
        audit(source, name, raw, label, model, language, families, counts, cases)

    report = {
        "summary": {key: dict(value) for key, value in counts.items()},
        "errors": cases,
        "contract": "Historical valid() prefix semantics for comparable 3086-sample accuracy. "
                    "strict_valid additionally reports full-document strict decodability. "
                    "Oracle is decoded-text equivalence, not codec-name identity.",
        "limitation": "This is an oracle upper bound conditional on existing candidate sets, "
                      "not achievable reranker accuracy. Small, correlated external sets "
                      "are diagnostic only. No training or release changes.",
    }
    output = Path("research-results/candidate-oracle")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print("CANDIDATE_ORACLE_SUMMARY", json.dumps(report["summary"]), flush=True)
    print("CANDIDATE_ORACLE_ERROR_COUNT", len(cases), flush=True)


if __name__ == "__main__":
    main()
