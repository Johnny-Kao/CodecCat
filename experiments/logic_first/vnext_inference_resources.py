"""Loaded-model inference-only resource profile on a pinned Linux runner.

Training happens in a separate process and is not counted as inference load.
This is an observation, NOT a self-assigned performance acceptance threshold.
"""
import argparse
import hashlib
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.vnext_decoding import decode


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def linux_rss_kib():
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def run(a):
    path = Path(a.model)
    raw_archive = path.read_bytes()
    manifest = json.loads(Path(a.manifest).read_text())
    profile = a.profile
    entry = manifest["profiles"][profile]
    if hashlib.sha256(raw_archive).hexdigest() != entry["sha256"]:
        raise ValueError("model archive differs from year-locked manifest")
    before_rss = linux_rss_kib()
    model = TrainableDetector.load_portable(path)
    after_rss = linux_rss_kib()
    if model.fingerprint != entry["fingerprint"]:
        raise ValueError("model training fingerprint does not match manifest")
    samples = {
        "short64": ("Technical prose 中文 日本語 Русский " * 5).encode("utf-8")[:64],
        "medium512": ("Multilingual real input Français 日本語 Русский مرحبا. " * 40).encode("utf-8")[:512],
        "long4096": ("A long English Unicode mixed sentence: 日本語 Ελληνικά العربية. " * 300).encode("utf-8")[:4096],
        "invalid512": (b"\xff\xfe\xfdABC" * 110)[:512],
        "utf16_512": ("UTF-16 input without a BOM, with repetitive words. " * 35).encode("utf-16-le")[:512],
    }
    rows = {}
    for key, data in samples.items():
        # Fixed input samples, no training or calibrated confidence.
        for _ in range(5):
            model.rank(data, final=len(data) < 4096)
        timings = []
        for _ in range(a.iterations):
            start = time.perf_counter_ns()
            model.rank(data, final=len(data) < 4096)
            timings.append(time.perf_counter_ns() - start)
        result = decode(model, data, bom_policy="preserve")
        rows[key] = {
            "n": a.iterations, "bytes": len(data),
            "p50_ms": round(percentile(timings, .50) / 1e6, 4),
            "p95_ms": round(percentile(timings, .95) / 1e6, 4),
            "predicted_encoding": result.encoding,
            "decoding_accepted": result.accepted,
        }
    report = {
        "profile": profile,
        "source_archive_sha256": entry["sha256"],
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "samples": rows,
        "process_rss_before_load_kib": before_rss,
        "process_rss_after_load_kib": after_rss,
        "incremental_loaded_rss_kib": None if before_rss is None or after_rss is None else after_rss - before_rss,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": "Ubuntu runner observations, samples synthetic, Python startup and imports contribute to baseline RSS; single runner cannot establish cross-architecture robustness.",
    }
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{profile}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("RESOURCE_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("RESOURCE_END")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--profile", choices=("stable23", "experimental13"), required=True)
    p.add_argument("--output", default="research-results/vnext-resources")
    p.add_argument("--iterations", type=int, default=120)
    run(p.parse_args())
