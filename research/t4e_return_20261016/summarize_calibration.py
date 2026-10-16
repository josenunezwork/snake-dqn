"""Summarize saved calibration updates and interpolate a shared initial weight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def geometric_mean(values: list[float]) -> float:
    """Require finite positive norm ratios before a log-space calibration."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array) & (array > 0)):
        raise ValueError("calibration requires finite positive ratios")
    return float(np.exp(np.mean(np.log(array))))


def summarize(root: Path) -> dict:
    """Use all 300 saved updates; never select by outcome or survival."""
    runs = []
    for path in sorted(root.glob("B8_s*_W*/calibration_receipt.json")):
        receipt = json.loads(path.read_text())
        summary = json.loads((path.parent / "summary.json").read_text())
        updates_path = path.parent / "calibration_updates.jsonl"
        updates = [json.loads(line) for line in updates_path.read_text().splitlines()]
        if receipt["trainer_exit"] or summary["halt"] or len(updates) != 300:
            raise ValueError(f"incomplete smoke: {path}")
        if summary["updates"] != 300 or summary["actor_exitcodes"] != [0, 0]:
            raise ValueError(f"update count or actor cleanup failed: {path}")
        ratios = [row["return_to_td_grad_ratio"] for row in updates]
        runs.append(
            {
                "run": path.parent.name,
                "seed": int(summary["args"]["seed"]),
                "weight": float(summary["args"]["return_weight"]),
                "device": receipt["device"],
                "updates": len(updates),
                "transitions": summary["transitions"],
                "first_ratio": ratios[0],
                "init_ratio_gmean_1_50": geometric_mean(ratios[:50]),
                "middle_ratio_gmean_126_175": geometric_mean(ratios[125:175]),
                "last_ratio_gmean_251_300": geometric_mean(ratios[-50:]),
                "max_abs_q": max(row["max_abs_q"] for row in updates),
                "batch_compositions": sorted(
                    {
                        (row["anchor_rows"], row["return_rows"], row["agent_td_rows"])
                        for row in updates
                    }
                ),
                "wall_seconds": summary["wall_seconds"],
                "halt": summary["halt"],
                "flags": summary["flags"],
                "actor_exitcodes": summary["actor_exitcodes"],
                "evidence_sha256": {
                    str(item): hashlib.sha256(item.read_bytes()).hexdigest()
                    for item in (path, updates_path, path.parent / "summary.json")
                },
            }
        )
    grid = []
    required = {0.01, 0.02, 0.05, 0.2, 1.0}
    weights = {row["weight"] for row in runs}
    if not required <= weights:
        raise ValueError("required five-weight grid is incomplete")
    for weight in sorted(weights):
        pair = [row for row in runs if row["weight"] == weight]
        if sorted(row["seed"] for row in pair) != [10, 12]:
            raise ValueError(f"missing or duplicate calibration pair at W={weight}")
        grid.append(
            {
                "weight": weight,
                "pooled_init_ratio": geometric_mean([row["init_ratio_gmean_1_50"] for row in pair]),
            }
        )
    brackets = [
        (a, b)
        for a, b in zip(grid, grid[1:])
        if a["pooled_init_ratio"] <= 0.5 <= b["pooled_init_ratio"]
    ]
    if not brackets:
        # The spec permits geometric interpolation, not extrapolation.
        raise ValueError(f"target 0.5 not bracketed by measured grid: {grid}")
    low, high = brackets[0]
    fraction = math.log(0.5 / low["pooled_init_ratio"]) / math.log(
        high["pooled_init_ratio"] / low["pooled_init_ratio"]
    )
    interpolated = math.exp(
        math.log(low["weight"]) + fraction * math.log(high["weight"] / low["weight"])
    )
    return {
        "schema": "t4e-calibration-summary/v1",
        "runs": runs,
        "grid": grid,
        "selection": "log-log interpolation of equally pooled per-seed first-50 geometric means",
        "target_ratio": 0.5,
        "interpolated_weight": interpolated,
        "chosen_weight": float(f"{interpolated:.2g}"),
    }


def main() -> None:
    """Write the complete auditable summary to a new file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"chosen_weight": result["chosen_weight"], "grid": result["grid"]}))


if __name__ == "__main__":
    main()
