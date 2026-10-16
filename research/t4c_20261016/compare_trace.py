"""Compare bounded default-path traces produced by the existing T4a harness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def compare(baseline: dict, current: dict) -> dict:
    """Require identical tensors, RNG and every pre-existing stable statistic."""
    assert baseline["modes"].keys() == current["modes"].keys()
    checks = {}
    for mode, before in baseline["modes"].items():
        after = current["modes"][mode]
        assert len(before["trace"]) == len(after["trace"]) == 8
        mismatches = []
        extra = set()
        for left, right in zip(before["trace"], after["trace"]):
            for field in (
                "online_sha256",
                "target_sha256",
                "online_tensor_sha256",
                "target_tensor_sha256",
                "rng_sha256",
                "torch_rng_sha256",
            ):
                if left[field] != right[field]:
                    mismatches.append([left["update"], field])
            for key, value in left["stable_stats"].items():
                if key not in right["stable_stats"] or value != right["stable_stats"][key]:
                    mismatches.append([left["update"], "stats." + key])
            extra.update(set(right["stable_stats"]) - set(left["stable_stats"]))
        checks[mode] = {
            "equal": not mismatches,
            "mismatches": mismatches,
            "new_telemetry_fields": sorted(extra),
            "updates": 8,
        }
    return {"pass": all(row["equal"] for row in checks.values()), "modes": checks}


def main() -> int:
    """Write a source-bound diagnostic receipt, not a training result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    current = json.loads(args.current.read_text())
    receipt = compare(baseline, current)
    root = Path(__file__).resolve().parents[2]
    receipt.update(
        {
            "schema": "t4c-default-trace/v1",
            "baseline_commit": baseline["commit"],
            "harness": "research/t4a_two_sided_20261016/default_trace.py",
            "cpu_only": True,
            "threads": 1,
            "trace_files": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.baseline, args.current)
            },
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    root / "research/redesign_m3c_20261008/learner_m3c.py",
                    root / "research/redesign_m3c_20261008/train_m3c.py",
                    root / "research/t3_rollout_labels_20261015/bank.py",
                )
            },
        }
    )
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
