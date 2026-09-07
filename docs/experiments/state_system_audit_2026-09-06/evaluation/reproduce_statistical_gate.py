#!/usr/bin/env python3
"""Reproduce minimum-seed and baseline-only pilot counterexamples."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.scripts.eval_stats import paired_stats, recommended_seed_count, sample_std  # noqa: E402
from src.scripts.tournament_eval import promotion_decision  # noqa: E402


def main() -> None:
    """Write deterministic arithmetic examples beside this script."""
    paired = paired_stats([11.0, 21.0], [10.0, 20.0])
    decision = promotion_decision(
        {"frozen": dict(paired), "scripted": dict(paired)},
        ["frozen", "scripted"],
    )

    baseline = [100.0, 100.0, 100.0, 100.0]
    candidate = [100.0, 100.0, 100.0, 300.0]
    deltas = [cand - base for cand, base in zip(candidate, baseline)]
    baseline_sd = sample_std(baseline)

    data = {
        "two_seed_zero_sample_variance": {
            "paired": paired,
            "decision_two_mixes": decision,
            "note": "Current gate has no CLI minimum beyond paired_stats n>=2.",
        },
        "baseline_only_pilot_counterexample": {
            "baseline": baseline,
            "candidate": candidate,
            "baseline_sd_used_by_pilot": baseline_sd,
            "actual_paired_delta_sd": sample_std(deltas),
            "recommended_n_from_baseline_sd_for_mde_3": recommended_seed_count(
                baseline_sd, 3.0
            ),
            "note": "A baseline-only sample cannot identify paired-delta variance.",
        },
    }
    output = Path(__file__).with_name("statistical_gate_repro.json")
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
