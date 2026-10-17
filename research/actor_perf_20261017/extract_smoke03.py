#!/usr/bin/env python3
"""Summarize saved smoke-03 receipts without launching or modifying a run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.root / "colab_throughput_results.json"
    report = json.loads(source.read_text())
    out = {"commit": report["commit"], "source_sha256": {}, "rows": []}
    out["source_sha256"][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for config in report["configs"]:
        name = config["name"]
        folder = args.root / name / "attempt-001"
        summary = json.loads((folder / "summary.json").read_text())
        telemetry, logs = [], []
        for filename, target in (("telemetry.jsonl", telemetry), ("log.jsonl", logs)):
            path = folder / filename
            for index, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "transitions" in record:
                    target.append({"line": index, "record": record})
        for filename in ("summary.json", "telemetry.jsonl", "log.jsonl"):
            path = folder / filename
            out["source_sha256"][str(path.relative_to(args.root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        last = telemetry[-1]["record"]
        # Aggregate deltas in the final approximately 60s telemetry window; no mean of noisy rates.
        first = min(telemetry, key=lambda r: abs(r["record"]["t"] - (last["t"] - 60)))
        initial = first["record"]
        seconds = last["t"] - initial["t"]
        delta = last["transitions"] - initial["transitions"]
        row = {
            "name": name,
            "resources": config.get("resources"),
            "summary": {
                k: summary.get(k)
                for k in ("transitions", "updates", "wall_seconds", "rate", "actor_exitcodes")
            },
            "args": {
                k: summary["args"].get(k)
                for k in (
                    "actors",
                    "worlds",
                    "batch_scale",
                    "gather_threads",
                    "compile_actor",
                    "actor_device",
                    "all_snakes_learner",
                    "update_every",
                )
            },
            "last_log": logs[-1],
            "last_telemetry": telemetry[-1],
            "window": {
                "first_line": first["line"],
                "last_line": telemetry[-1]["line"],
                "seconds": seconds,
                "transitions": delta,
                "transitions_per_second": delta / seconds,
            },
        }
        out["rows"].append(row)
    learner_path = args.root / "p0_fixed/bench_raw-rings_g3.json"
    if learner_path.exists():
        out["source_sha256"][str(learner_path.relative_to(args.root))] = hashlib.sha256(
            learner_path.read_bytes()
        ).hexdigest()
        out["learner_raw_g3_receipt"] = json.loads(learner_path.read_text())
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
