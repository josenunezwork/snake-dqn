#!/usr/bin/env python3
"""Observation-distribution drift check between two environment configurations.

Blueprint §5.3: the standing train/serve drift instrument. The known failure
mode (Part I, finding 2) is that Apex actors train on a 0.2-scale board while
eval/serving runs full-size, so danger/wall/board-relative features are drawn
from different distributions. This tool makes that drift measurable in CI:

1. Build two observation samples — config A vs config B (e.g. the actor-scale
   env vs the full-size eval env) — by stepping real games with random-SAFE
   actions and collecting the real ``get_state`` vectors.
2. Compute the per-dimension two-sample Kolmogorov-Smirnov statistic plus a
   histogram summary of both samples.
3. Exit non-zero if any dimension's KS statistic exceeds the threshold
   (default 0.25), and emit markdown + JSON reports.

``--self-test`` exercises the statistics pipeline on synthetic arrays (no
checkpoints, no game) so the math is unit-testable and CI-verifiable anywhere.

Usage:
  SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/obs_histogram_diff.py \
    --config-a configs/default.yaml --scale-a 0.2 \
    --config-b configs/default.yaml --scale-b 1.0 \
    --frames 400 --threshold 0.25 \
    --json-output logs/obs_drift.json --md-output logs/obs_drift.md

  ./venv/bin/python src/scripts/obs_histogram_diff.py --self-test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEFAULT_THRESHOLD = 0.25

# Known 58/61-D feature layout (see src/game/snake_state.py); used to
# label report rows. Anything outside these ranges falls back to "dim_<i>".
_FEATURE_RANGES = (
    (0, 4, "direction"),
    (4, 5, "length"),
    (5, 7, "food_xy"),
    (7, 8, "food_dist"),
    (8, 24, "food_density"),
    (24, 40, "danger_map"),
    (40, 44, "boundary"),
    (44, 47, "enemy1"),
    (47, 49, "enemy1_heading"),
    (49, 50, "enemy1_trend"),
    (50, 53, "enemy2"),
    (53, 54, "kill_opp"),
    (54, 57, "action_danger"),
    (57, 58, "boost_avail"),
    (58, 61, "free_space"),
)


def feature_label(dim: int) -> str:
    """Human label for a state dimension in the known 58/61-D layout."""
    for start, end, name in _FEATURE_RANGES:
        if start <= dim < end:
            return f"{name}[{dim - start}]" if end - start > 1 else name
    return f"dim_{dim}"


# =============================================================================
# Pure statistics (unit-testable without the game)
# =============================================================================
def ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max ECDF distance).

    Args:
        sample_a: 1-D sample from distribution A.
        sample_b: 1-D sample from distribution B.

    Returns:
        ``sup_x |ECDF_A(x) - ECDF_B(x)|`` in [0, 1].
    """
    a = np.sort(np.asarray(sample_a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(sample_b, dtype=np.float64).ravel())
    if a.size == 0 or b.size == 0:
        raise ValueError("KS statistic requires non-empty samples")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def histogram_summary(sample: np.ndarray) -> Dict[str, float]:
    """Summary statistics for one dimension's sample (mean/std/quantiles)."""
    x = np.asarray(sample, dtype=np.float64).ravel()
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def compare_observation_sets(
    obs_a: np.ndarray,
    obs_b: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Per-dimension KS comparison of two observation matrices.

    Args:
        obs_a: Sample A, shape ``(n_a, dims)``.
        obs_b: Sample B, shape ``(n_b, dims)`` — same ``dims`` as A.
        threshold: KS value above which a dimension is flagged as drifted.
        labels: Optional per-dimension labels (defaults to the feature layout).

    Returns:
        Dict with per-dim results, ``max_ks``, ``flagged`` dims, and ``passed``.
    """
    a = np.asarray(obs_a, dtype=np.float64)
    b = np.asarray(obs_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Observation sets must be 2-D (samples x dims)")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Dimension mismatch: A has {a.shape[1]} dims, B has {b.shape[1]}")

    dims: List[Dict[str, Any]] = []
    flagged: List[int] = []
    for dim in range(a.shape[1]):
        ks = ks_statistic(a[:, dim], b[:, dim])
        over = ks > threshold
        if over:
            flagged.append(dim)
        label = labels[dim] if labels is not None else feature_label(dim)
        dims.append(
            {
                "dim": dim,
                "label": label,
                "ks": ks,
                "flagged": over,
                "a": histogram_summary(a[:, dim]),
                "b": histogram_summary(b[:, dim]),
            }
        )
    return {
        "threshold": float(threshold),
        "n_a": int(a.shape[0]),
        "n_b": int(b.shape[0]),
        "num_dims": int(a.shape[1]),
        "max_ks": max(d["ks"] for d in dims),
        "flagged": flagged,
        "passed": not flagged,
        "dims": dims,
    }


def markdown_report(result: Dict[str, Any], label_a: str, label_b: str) -> str:
    """Render a comparison result as a markdown report (flagged dims first)."""
    lines = [
        "# Observation distribution drift: KS check",
        "",
        f"- A: **{label_a}** ({result['n_a']} samples)",
        f"- B: **{label_b}** ({result['n_b']} samples)",
        f"- Threshold: KS > {result['threshold']:.2f}",
        f"- Result: **{'PASS' if result['passed'] else 'DRIFT DETECTED'}** "
        f"(max KS = {result['max_ks']:.3f}, {len(result['flagged'])}/{result['num_dims']} "
        "dims flagged)",
        "",
        "| dim | feature | KS | drift | A mean±std | A [p05, p95] | B mean±std | B [p05, p95] |",
        "|----:|:--------|---:|:-----:|:-----------|:-------------|:-----------|:-------------|",
    ]
    ordered = sorted(result["dims"], key=lambda d: d["ks"], reverse=True)
    for d in ordered:
        a, b = d["a"], d["b"]
        lines.append(
            f"| {d['dim']} | {d['label']} | {d['ks']:.3f} | {'YES' if d['flagged'] else ''} "
            f"| {a['mean']:.3f}±{a['std']:.3f} | [{a['p05']:.3f}, {a['p95']:.3f}] "
            f"| {b['mean']:.3f}±{b['std']:.3f} | [{b['p05']:.3f}, {b['p95']:.3f}] |"
        )
    return "\n".join(lines)


# =============================================================================
# Observation collection (the only game-touching code; imported lazily)
# =============================================================================
def collect_observations(
    config_path: str,
    board_scale: float,
    frames: int,
    warmup: int = 30,
    num_snakes: int = 6,
    seed: int = 0,
) -> np.ndarray:
    """Collect real ``get_state`` observations from random-safe stepping.

    Builds a headless game under the given config/board scale, sets every
    snake to pure random-safe exploration (epsilon 1.0, no danger/boost
    exploration bias — the policy network is never consulted), and records
    each alive snake's observation vector every post-warmup frame. Dead
    snakes respawn so the sample is not survivor-censored.

    Args:
        config_path: YAML config applied to the global GameConfig.
        board_scale: Board scale multiplier (0.2 = actor-scale, 1.0 = full).
        frames: Number of post-warmup frames to record.
        warmup: Frames stepped (not recorded) before collection starts.
        num_snakes: Snakes in the arena.
        seed: RNG seed for random/numpy/torch.

    Returns:
        Array of shape ``(num_samples, state_dims)``.
    """
    from src.core.config_loader import load_and_initialize_config
    from src.game.game_state import GameState

    load_and_initialize_config(config_path)

    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)

    game_state = GameState(
        headless=True,
        snake_policies=["apex"] * num_snakes,
        num_snakes=num_snakes,
        board_scale=board_scale,
    )
    shared = getattr(game_state, "_shared_policy", None)
    if shared is not None and hasattr(shared, "training"):
        shared.training = False  # no replay writes
    game_state._shared_policy = None  # no train_step
    for snake in game_state.snakes:
        if hasattr(snake, "actor_epsilon"):
            snake.actor_epsilon = 1.0  # always explore: random.choice(safe_actions)
        if hasattr(snake, "danger_exploration_rate"):
            snake.danger_exploration_rate = 0.0
        if hasattr(snake, "boost_exploration_rate"):
            snake.boost_exploration_rate = 0.0

    rows: List[np.ndarray] = []
    for frame in range(warmup + frames):
        game_state.update(train_mode=True, learn=False, allow_respawn=True)
        if frame < warmup:
            continue
        for snake in game_state.snakes:
            state = getattr(snake, "last_state", None)
            if snake.is_alive and state is not None:
                rows.append(np.asarray(state.detach().cpu().numpy(), dtype=np.float64).ravel())
    game_state.full_cleanup()

    if not rows:
        raise RuntimeError(f"Collected no observations from {config_path} (scale {board_scale})")
    return np.stack(rows)


# =============================================================================
# Self-test (synthetic arrays; no game, no checkpoints)
# =============================================================================
def self_test(threshold: float = DEFAULT_THRESHOLD) -> int:
    """Verify the KS pipeline on synthetic data: 0 on success, 1 on failure.

    Checks that (a) two draws from the same distribution pass, and (b) a
    single shifted dimension is flagged — and only that dimension.
    """
    rng = np.random.default_rng(12345)
    n, dims, shifted_dim = 4000, 6, 3
    base_a = rng.normal(0.0, 1.0, size=(n, dims))
    base_b = rng.normal(0.0, 1.0, size=(n, dims))

    same = compare_observation_sets(base_a, base_b, threshold=threshold)
    if not same["passed"]:
        print(f"self-test FAIL: identical distributions flagged {same['flagged']}")
        return 1

    drifted = base_b.copy()
    drifted[:, shifted_dim] += 1.0  # a full sigma shift: KS ~ 0.38 >> threshold
    diff = compare_observation_sets(base_a, drifted, threshold=threshold)
    if diff["flagged"] != [shifted_dim]:
        print(f"self-test FAIL: expected flagged=[{shifted_dim}], got {diff['flagged']}")
        return 1

    disjoint = ks_statistic(np.zeros(100), np.ones(100))
    if not np.isclose(disjoint, 1.0):
        print(f"self-test FAIL: disjoint samples KS={disjoint}, expected 1.0")
        return 1

    print(
        "self-test PASS: "
        f"same-dist max KS={same['max_ks']:.3f} (no flags), "
        f"shifted dim {shifted_dim} KS={diff['dims'][shifted_dim]['ks']:.3f} (flagged), "
        "disjoint KS=1.0"
    )
    return 0


# =============================================================================
# CLI
# =============================================================================
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-a", default="configs/default.yaml")
    parser.add_argument("--config-b", default="configs/default.yaml")
    parser.add_argument(
        "--scale-a", type=float, default=0.2, help="Board scale for sample A (actor default)"
    )
    parser.add_argument(
        "--scale-b", type=float, default=1.0, help="Board scale for sample B (eval/serve default)"
    )
    parser.add_argument("--frames", type=int, default=400, help="Recorded frames per sample")
    parser.add_argument("--warmup", type=int, default=30, help="Unrecorded warmup frames")
    parser.add_argument("--num-snakes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--md-output", default=None)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the statistics pipeline on synthetic arrays and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test(threshold=args.threshold)

    label_a = f"{args.config_a} @ scale {args.scale_a}"
    label_b = f"{args.config_b} @ scale {args.scale_b}"
    obs_a = collect_observations(
        args.config_a, args.scale_a, args.frames, args.warmup, args.num_snakes, args.seed
    )
    obs_b = collect_observations(
        args.config_b, args.scale_b, args.frames, args.warmup, args.num_snakes, args.seed
    )

    result = compare_observation_sets(obs_a, obs_b, threshold=args.threshold)
    report = markdown_report(result, label_a, label_b)
    print(report)

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"a": label_a, "b": label_b, **result}, indent=2))
        print(f"\nWrote {out}")
    if args.md_output:
        out = Path(args.md_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n")
        print(f"Wrote {out}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
