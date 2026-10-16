"""Summarize completed RETURN_C contrast calibration smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def geometric_mean(values: list[float]) -> float:
    """Return a finite positive geometric mean for gradient-ratio windows."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array) & (array > 0)):
        raise ValueError("calibration requires finite positive ratios")
    return float(np.exp(np.mean(np.log(array))))


def summarize(root: Path) -> dict:
    """Require complete, clean paired 300-update smokes before choosing W."""
    runs = []
    for receipt_path in sorted(root.glob("B8_s*_W*/calibration_receipt.json")):
        run = receipt_path.parent
        receipt, summary = json.loads(receipt_path.read_text()), json.loads(
            (run / "summary.json").read_text()
        )
        updates_path = run / "calibration_updates.jsonl"
        updates = [json.loads(line) for line in updates_path.read_text().splitlines()]
        if (
            receipt["device"] != "mps"
            or receipt["trainer_exit"]
            or summary["halt"]
            or len(updates) != 300
        ):
            raise ValueError(f"incomplete or non-MPS smoke: {receipt_path}")
        if summary["updates"] != 300 or summary["actor_exitcodes"] != [0, 0]:
            raise ValueError(f"update count or actor cleanup failed: {receipt_path}")
        ratios = [row["return_to_td_grad_ratio"] for row in updates]
        runs.append(
            {
                "run": run.name,
                "seed": int(summary["args"]["seed"]),
                "weight": float(summary["args"]["return_weight"]),
                "device": "mps",
                "updates": len(updates),
                "init_ratio_gmean_1_50": geometric_mean(ratios[:50]),
                "middle_ratio_gmean_126_175": geometric_mean(ratios[125:175]),
                "last_ratio_gmean_251_300": geometric_mean(ratios[-50:]),
                "max_abs_q": max(row["max_abs_q"] for row in updates),
                "active_fraction_mean_1_50": float(
                    np.mean([row["return_contrast_active_fraction"] for row in updates[:50]])
                ),
                "active_fraction_mean_all": float(
                    np.mean([row["return_contrast_active_fraction"] for row in updates])
                ),
                "batch_compositions": sorted(
                    {
                        (row["anchor_rows"], row["return_rows"], row["agent_td_rows"])
                        for row in updates
                    }
                ),
                "evidence_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (receipt_path, updates_path, run / "summary.json")
                },
            }
        )
    grid = []
    for weight in sorted({row["weight"] for row in runs}):
        pair = [row for row in runs if row["weight"] == weight]
        if sorted(row["seed"] for row in pair) != [10, 12]:
            raise ValueError(f"missing or duplicate calibration pair at W={weight}")
        grid.append(
            {
                "weight": weight,
                "pooled_init_ratio": geometric_mean([row["init_ratio_gmean_1_50"] for row in pair]),
            }
        )
    if len(grid) < 2:
        raise ValueError("at least two calibrated weights are required to bracket target 0.5")
    brackets = [
        (a, b)
        for a, b in zip(grid, grid[1:])
        if a["pooled_init_ratio"] <= 0.5 <= b["pooled_init_ratio"]
    ]
    if not brackets:
        raise ValueError(f"target 0.5 not bracketed by measured grid: {grid}")
    low, high = brackets[0]
    fraction = math.log(0.5 / low["pooled_init_ratio"]) / math.log(
        high["pooled_init_ratio"] / low["pooled_init_ratio"]
    )
    interpolated = math.exp(
        math.log(low["weight"]) + fraction * math.log(high["weight"] / low["weight"])
    )
    return {
        "schema": "t4f-contrast-calibration-summary/v1",
        "runs": runs,
        "grid": grid,
        "selection": "log-log interpolation of equally pooled per-seed first-50 geometric means",
        "target_ratio": 0.5,
        "interpolated_weight": interpolated,
        "chosen_weight": float(f"{interpolated:.2g}"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with args.out.open("x") as stream:
        json.dump(summarize(args.root), stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
