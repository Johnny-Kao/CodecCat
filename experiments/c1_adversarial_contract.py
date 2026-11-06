"""Adversarial controls and contract profiling for C1 reranking, research only.

No change to installed CodecCat or its model; source-free deterministic bytes.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

import codeccat
from western_c1_counterfactual import rerank_control

OUT = Path("research-results/c1-adversarial-contract.json")
# Unicode C1 is legal in Latin-1. The payloads are deliberately constructed
# to test whether a heuristic can ever distinguish it from CP1252 punctuation.
TEMPLATES = (
    "An international invoice records supplier quantities, delivery dates, "
    "freight totals, and transaction details for the accounting office. ",
    "Les documents administratifs contiennent le nom du destinataire, "
    "la date de réception et les références de la facture. ",
    "Beschreibung der Artikel und Bedingungen für die Auslieferung "
    "einschließlich Bestellnummer, Preis und Kundendienst. ",
    "The engineering report describes signal control interfaces, "
    "transport telemetry, diagnostic packets, and process monitoring. ",
)
CONTROL_BYTES = (0x80, 0x81, 0x82, 0x85, 0x8a, 0x91, 0x92, 0x93, 0x94, 0x96, 0x9f)
POSITIONS = ("head", "middle", "tail")


def create_cases():
    cases = []
    for template_index, text in enumerate(TEMPLATES):
        # Three positions and three sizes; no random seed or selection by outcomes.
        base = text.encode("latin-1")
        for repetition in (1, 3, 8):
            passage = base * repetition
            for ch in CONTROL_BYTES:
                for position in POSITIONS:
                    pos = {"head": 0, "middle": len(passage)//2,
                           "tail": len(passage)}[position]
                    raw = passage[:pos] + bytes([ch]) + passage[pos:]
                    # Both the ISO-8859-1 semantics and CP1252 alternatives
                    # are explicit, even for 0x81 (CP1252 rejects).
                    expected = raw.decode("iso-8859-1", "strict")
                    cases.append((template_index, repetition, ch, position, raw, expected))
    return cases


def measure(detector, samples, repeats=2):
    baseline_times, rule_times = [], []
    for raw in samples:
        detected = detector.detect(raw)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            _ = detected.text
            baseline_times.append(time.perf_counter_ns() - start)
            start = time.perf_counter_ns()
            rerank_control(raw, detected)
            rule_times.append(time.perf_counter_ns() - start)
    return {
        "baseline_postprocess_median_ns": statistics.median(baseline_times),
        "counterfactual_postprocess_median_ns": statistics.median(rule_times),
        "note": "Measures output postprocessing only, not detector inference; use neither as a platform performance claim.",
    }


def main():
    assert importlib.metadata.version("codeccat") == "0.1.0"
    detector = codeccat.CodecCat()
    cases = create_cases()
    counters = Counter()
    by_byte = defaultdict(Counter)
    by_size = defaultdict(Counter)
    examples = []
    for template, repetition, control, position, raw, truth in cases:
        detection = detector.detect(raw)
        after, route = rerank_control(raw, detection)
        before_correct = detection.text == truth
        after_correct = after == truth
        counters["n"] += 1
        counters["baseline_correct"] += int(before_correct)
        counters["candidate_correct"] += int(after_correct)
        counters["harmed"] += int(before_correct and not after_correct)
        counters["helped"] += int(not before_correct and after_correct)
        counters["changed"] += int(detection.text != after)
        counters["c1_present_in_reference"] += int(any(0x80 <= ord(c) <= 0x9f for c in truth))
        counters["baseline_uncertain"] += int(bool(detection.ambiguous))
        for subset in (by_byte[str(control)], by_size[str(repetition)]):
            subset["n"] += 1
            subset["baseline_correct"] += int(before_correct)
            subset["candidate_correct"] += int(after_correct)
            subset["harmed"] += int(before_correct and not after_correct)
            subset["changed"] += int(detection.text != after)
        if before_correct and not after_correct:
            examples.append({
                "template_index": template, "repetition": repetition,
                "control_byte_hex": f"0x{control:02x}", "position": position,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "baseline_codec": detection.encoding,
                "model_candidates": list(detection.candidates),
                "baseline_ambiguous": detection.ambiguous,
                "rule_action": route,
            })
    report = {
        "schema": "codeccat-c1-adversarial-contract-v1",
        "status": "research_only",
        "source": "deliberate original Latin-1 control characters; synthetic counterexamples",
        "outcomes": dict(counters),
        "by_control_byte": {k: dict(v) for k,v in sorted(by_byte.items())},
        "by_repetition": {k: dict(v) for k,v in sorted(by_size.items())},
        "harmed_metadata": examples[:100],
        "timing": measure(detector, [c[4] for c in cases[:72]]),
        "raw_content_exported": False,
        "limitation": "Designed to expose worst-case correct Latin-1 control input, not prevalence or real-world accuracy.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("C1_ADVERSARIAL_CONTRACT", json.dumps({
        "n": counters["n"], "baseline_correct": counters["baseline_correct"],
        "candidate_correct": counters["candidate_correct"],
        "harmed": counters["harmed"], "helped": counters["helped"],
        "changed": counters["changed"], "baseline_uncertain": counters["baseline_uncertain"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
