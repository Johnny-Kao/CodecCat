"""Evaluate an architecture concept without refitting or relabeling old data.

This operates on pinned paired predictions from ACTIONS #41233450603.
Aim: isolate specialist corrections to *uncertain Latin-family decisions*,
leaving all unrelated baseline predictions unchanged by design.

Ablations are DEVELOPMENT ONLY: this heldout has already been inspected.
"""
from __future__ import annotations
import argparse
import codecs
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

PAIRED_SHA = "4be6f883b17495490a992f905dc149d4a89c22bbc2d3e3b67919047d33c71a8c"
PAIRS = frozenset({
    frozenset(("iso8859-16", "iso8859-2")),
    frozenset(("iso8859-2", "cp1250")),
    frozenset(("cp1252", "iso8859-1")),
    frozenset(("cp1252", "iso8859-15")),
})


def norm(label):
    if label is None:
        return None
    try:
        return codecs.lookup(label).name
    except LookupError:
        return None


def latin(label):
    x = norm(label)
    return bool(x and (x.startswith("iso8859-") or x.startswith("cp12")))


def allow(baseline, candidate, rule):
    a, b = norm(baseline["pred"]), norm(candidate["pred"])
    if not a or not b or a == b:
        return False
    if not latin(a) or not latin(b):
        return False
    return rule == "latin-family" or frozenset((a, b)) in PAIRS


def report(rows):
    result = {"n": len(rows), "rules": {}}
    for rule in ("latin-family", "observed-confusion-pairs"):
        decisions = []
        for row in rows:
            b = row["predictions"]["baseline"]
            c = row["predictions"]["latin_only"]
            selected = c if allow(b, c, rule) else b
            decisions.append({"truth": row["truth"], "baseline": b,
                              "outcome": selected,
                              "changed": selected is c and c["pred"] != b["pred"]})
        orig = sum(d["baseline"]["text"] for d in decisions)
        selected = sum(d["outcome"]["text"] for d in decisions)
        helped = sum(d["outcome"]["text"] and not d["baseline"]["text"] for d in decisions)
        harmed = sum(d["baseline"]["text"] and not d["outcome"]["text"] for d in decisions)
        by_family = defaultdict(lambda: {"n":0,"old":0,"routed":0,"changed":0})
        for d in decisions:
            name = norm(d["truth"]) or "unknown"
            key = ("iso8859" if name.startswith("iso8859-") else
                   "win125x" if name.startswith("cp12") else
                   "unicode" if name.startswith("utf-") else
                   "other")
            stat = by_family[key]
            stat["n"] += 1
            stat["old"] += d["baseline"]["text"]
            stat["routed"] += d["outcome"]["text"]
            stat["changed"] += d["changed"]
        result["rules"][rule] = {
            "original_correct_text": orig,
            "routed_correct_text": selected,
            "corrected": helped, "harmed": harmed,"net":selected-orig,
            "changed_outputs":sum(d["changed"] for d in decisions),
            "families":dict(sorted(by_family.items())),
        }
    return result


def main(args):
    raw=Path(args.paired).read_bytes()
    assert sha256(raw).hexdigest() == PAIRED_SHA, "pinned source predictions do not match"
    rows=[json.loads(x) for x in raw.splitlines()]
    if len(rows)!=3086 or len({r["sha"] for r in rows})!=len(rows):
        raise ValueError("unexpected prior audit rows")
    result=report(rows)
    result["schema"]="codeccat-conditional-latin-specialist-development-v1"
    result["source"]="paired predictions from immutable Actions #41233450603"
    result["note"]="This reuses previously inspected development rows: not independent validation."
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("CONDITIONAL_SPECIALIST_SCREEN_BEGIN")
    print(json.dumps(result,sort_keys=True,indent=2))
    print("CONDITIONAL_SPECIALIST_SCREEN_END")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--paired",required=True)
    p.add_argument("--out",default="research-results/conditional-specialist")
    main(p.parse_args())
