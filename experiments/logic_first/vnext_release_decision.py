"""Machine-readable integrated go/no-go verdict, separate from CI success.

Passing a job means evidence was generated; it does not authorize a release.
Thresholds are comparative, not tuned to this benchmark. Explicitly
unverified gates remain false until independent evidence is provided.
"""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def evaluate(a):
    icu = load(a.icu_report)
    annual = load(a.annual_manifest)
    stable = load(a.stable_resource)
    trial = load(a.experimental_resource)
    if icu["fixture_commit"] != "ad110a2381f1b75bfb875fb0b672b7c38dce4e4b":
        raise ValueError("unexpected confirmation source commit")
    if icu["training_commit"] != annual["training_commit"]:
        raise ValueError("model and blind evaluation used different training snapshots")
    if annual["release_authorized"] is not False:
        raise ValueError("candidate training improperly asserts release authority")
    for report, name in ((stable, "stable23"), (trial, "experimental13")):
        if report["profile"] != name:
            raise ValueError("incorrect loaded-model resource profile")
        if report["source_archive_sha256"] != annual["profiles"][name]["sha256"]:
            raise ValueError("benchmark archive does not match trained model")

    m = icu["metrics"]
    chardet = m["chardet_7"]["same_text"]
    gates = {
        "bom_contract_and_inference_tests": {
            "passed": True,
            "basis": "Workflow runs full vNext pytest suite before this decision step.",
        },
        "immutable_training_model_lineage": {
            "passed": True,
            "basis": "Manifest pinned checkout SHA, archive SHA256 and independent resource process verification.",
        },
        "fresh_source_benchmark_meets_chardet": {
            "passed": m["stable23"]["same_text"] >= chardet,
            "basis": "On previously untouched ICU test source, conservative stable23 text-hit count must match or exceed chardet 7.6.0 before proposed stable default can pass.",
            "stable23": m["stable23"]["same_text"],
            "chardet": chardet,
            "n": icu["accepted_samples"],
        },
        "experimental_profile_improves_fresh_source": {
            "passed": m["experimental13"]["same_text"] >= m["stable23"]["same_text"],
            "basis": "No candidate promotion if held-out source has net decoded-text regression versus stable23.",
            "experimental13": m["experimental13"]["same_text"],
            "stable23": m["stable23"]["same_text"],
        },
        "language_coherence_L3_evidence": {
            "passed": False,
            "basis": "A trained L3 verifier with cross-source help/harm controls is not implemented or independently accepted.",
        },
        "available_arm64_runtime_validation": {
            "passed": False,
            "basis": "Only free private-repository Ubuntu x86_64 runner has been verified. Do not claim ARM64 results.",
        },
        "explicit_resource_budget_approved": {
            "passed": False,
            "basis": "Inference-only p50/p95 and RSS recorded on x86_64, but no independent production resource budgets have been approved.",
            "stable23_incremental_rss_kib": stable["incremental_loaded_rss_kib"],
            "experimental13_incremental_rss_kib": trial["incremental_loaded_rss_kib"],
        },
        "annual_retraining_recipe_reproducible": {
            "passed": bool(annual["profiles"].get("stable23") and annual["profiles"].get("experimental13")
                           and annual.get("dataset_fingerprint") and annual.get("training_commit")),
            "basis": "Current-year frozen corpus, immutable checkout and saved portable bundles verified. Future yearly source refresh is a separate operational obligation.",
        },
    }
    blocks = [k for k, v in gates.items() if not v["passed"]]
    result = {
        "schema": "vnext-release-decision-v1",
        "status": "GO" if not blocks else "NO_GO",
        "candidate_default": "stable23",
        "model_production_promoted": False,
        "model_released": False,
        "checklist": gates,
        "blocking_gates": blocks,
        "future_year_obligation": "Before each future annual retrain, freeze approved new training and untouched holdout snapshots; changing the year number alone is insufficient.",
        "independent_source": "unicode-org/icu",
        "independent_source_commit": icu["fixture_commit"],
        "explanation": "CI success means this evaluation executed correctly. Only zero blockers can authorize an owner signoff request; even then production deployment requires separate explicit authorization.",
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("FINAL_VERDICT_BEGIN")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("FINAL_VERDICT_END")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--icu-report", required=True)
    p.add_argument("--annual-manifest", required=True)
    p.add_argument("--stable-resource", required=True)
    p.add_argument("--experimental-resource", required=True)
    p.add_argument("--output", default="research-results/vnext-final-e2e/verdict.json")
    evaluate(p.parse_args())
