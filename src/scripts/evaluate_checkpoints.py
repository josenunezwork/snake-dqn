#!/usr/bin/env python3
"""Evaluate Apex checkpoints with deterministic greedy headless rollouts."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

# Add project root to path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import (  # noqa: E402
    apply_config_to_game_config,
    load_config,
)
from src.core.game_config import CheckpointSettings, initialize_config  # noqa: E402
from src.main import (  # noqa: E402
    apply_training_batch_size_override,
    run_learning_health_smoke,
)

SmokeRunner = Callable[..., Dict[str, Any]]


def parse_seed_list(value: str) -> List[int]:
    """Parse a comma-separated seed list."""
    seeds = []
    for raw_seed in value.split(","):
        raw_seed = raw_seed.strip()
        if not raw_seed:
            continue
        seeds.append(int(raw_seed))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or zero for empty values."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _set_eval_seed(seed: int) -> None:
    """Reset random generators before one deterministic rollout."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def summarize_rollouts(checkpoint: str, rollouts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed health-smoke results into one checkpoint score."""
    rewards = [float(run["episode"]["reward"]) for run in rollouts]
    foods = [int(run["episode"]["food_eaten"]) for run in rollouts]
    deaths = [int(run["episode"]["deaths"]) for run in rollouts]
    lengths = [int(run["episode"]["length"]) for run in rollouts]
    kills = [int(run["episode"]["kills"]) for run in rollouts]

    return {
        "checkpoint": checkpoint,
        "avg_reward": _mean(rewards),
        "avg_food": _mean([float(food) for food in foods]),
        "avg_length": _mean([float(length) for length in lengths]),
        "avg_deaths": _mean([float(death) for death in deaths]),
        "avg_kills": _mean([float(kill) for kill in kills]),
        "rewards": rewards,
        "food": foods,
        "deaths": deaths,
        "lengths": lengths,
        "kills": kills,
        "rollouts": list(rollouts),
    }


def checkpoint_rank_key(summary: Dict[str, Any]) -> tuple[float, float, float, float]:
    """Return the sort key for choosing the best greedy gameplay checkpoint."""
    return (
        float(summary["avg_reward"]),
        float(summary["avg_food"]),
        -float(summary["avg_deaths"]),
        float(summary["avg_length"]),
    )


def evaluate_checkpoint(
    checkpoint: str,
    *,
    frames: int,
    seeds: Iterable[int],
    smoke_runner: SmokeRunner = run_learning_health_smoke,
) -> Dict[str, Any]:
    """Evaluate one checkpoint across seeded greedy rollouts."""
    rollouts = []
    for seed in seeds:
        _set_eval_seed(int(seed))
        stats = smoke_runner(
            max_frames=frames,
            checkpoint_filename=None,
            checkpoint_path=checkpoint,
            eval_mode=True,
        )
        rollouts.append(stats)
    return summarize_rollouts(checkpoint, rollouts)


def evaluate_checkpoints(
    checkpoints: Sequence[str],
    *,
    frames: int,
    seeds: Sequence[int],
    smoke_runner: SmokeRunner = run_learning_health_smoke,
) -> List[Dict[str, Any]]:
    """Evaluate and rank checkpoints from best to worst."""
    summaries = [
        evaluate_checkpoint(
            checkpoint,
            frames=frames,
            seeds=seeds,
            smoke_runner=smoke_runner,
        )
        for checkpoint in checkpoints
    ]
    return sorted(summaries, key=checkpoint_rank_key, reverse=True)


def format_summary_table(summaries: Sequence[Dict[str, Any]]) -> str:
    """Format checkpoint scores for console output."""
    lines = [
        "rank | avg_reward | avg_food | avg_deaths | avg_length | checkpoint",
        "-----|------------|----------|------------|------------|-----------",
    ]
    for rank, summary in enumerate(summaries, start=1):
        lines.append(
            f"{rank:>4} | "
            f"{summary['avg_reward']:>10.2f} | "
            f"{summary['avg_food']:>8.1f} | "
            f"{summary['avg_deaths']:>10.2f} | "
            f"{summary['avg_length']:>10.1f} | "
            f"{summary['checkpoint']}"
        )
    return "\n".join(lines)


def configure_project(config_path: str, batch_size: Optional[int], checkpoint_dir: str) -> None:
    """Load YAML config and redirect probe artifacts away from saved models."""
    config = load_config(config_path)
    config = replace(
        config,
        checkpoint=CheckpointSettings(
            checkpoint_dir=checkpoint_dir,
            best_model_name=config.checkpoint.best_model_name,
        ),
    )
    initialize_config(config)
    apply_config_to_game_config(config)
    apply_training_batch_size_override(batch_size)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Apex checkpoints with greedy headless rollouts.",
    )
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint files to evaluate")
    parser.add_argument("--config", default="configs/training_fast.yaml", help="YAML config path")
    parser.add_argument("--frames", type=int, default=1000, help="Frames per rollout")
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=parse_seed_list("0,1,2"),
        help="Comma-separated rollout seeds",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch override")
    parser.add_argument(
        "--checkpoint-dir",
        default="/tmp/snake_dqn_eval_ckpts",
        help="Temporary checkpoint directory for eval helpers",
    )
    parser.add_argument("--json-output", default=None, help="Optional JSON summary path")
    parser.add_argument("--copy-best-to", default=None, help="Copy top checkpoint to this path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run checkpoint evaluation from the command line."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames <= 0:
        parser.error("--frames must be positive")

    configure_project(args.config, args.batch_size, args.checkpoint_dir)
    summaries = evaluate_checkpoints(args.checkpoints, frames=args.frames, seeds=args.seeds)

    print(format_summary_table(summaries))
    best = summaries[0]
    print(f"\nBest checkpoint: {best['checkpoint']}")

    if args.copy_best_to:
        target = Path(args.copy_best_to)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["checkpoint"], target)
        print(f"Copied best checkpoint to: {target}")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"Wrote JSON summary to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
