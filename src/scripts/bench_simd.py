"""Throughput benchmark for the NumPy batch simulator (blueprint P2 exit gate).

Steps ``src.simd_env.BatchSim`` over a batch of ``E`` environments x ``S`` snakes
for ``N`` frames on CPU and reports agent-steps/s and env-steps/s. Per-agent
action masks ARE included in the measured loop (they are computed inside
:meth:`BatchSim.step`); the raster featurizer is NOT yet included, so these
numbers are an upper bound on the dynamics half of a full env step.

Reference points (blueprint §0.10 / audit):
  - Audit baseline: ~2.1k transitions/s per single Python actor (live game).
  - P2 exit gate: >=40k agent-steps/s single process on this Mac (incl. masks).

Usage:
    ./venv/bin/python -m src.scripts.bench_simd
    ./venv/bin/python -m src.scripts.bench_simd --envs 256 --snakes 8 --steps 200
    ./venv/bin/python -m src.scripts.bench_simd --sweep
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.simd_env.batch_sim import BatchSim, BatchSimConfig

# Gate + baseline for context in the printed report.
GATE_AGENT_STEPS_PER_S = 40_000
AUDIT_ACTOR_TRANSITIONS_PER_S = 2_100


@dataclass(frozen=True)
class BenchResult:
    """One measured (E, S) benchmark point."""

    num_envs: int
    num_snakes: int
    num_steps: int
    warmup_steps: int
    wall_s: float
    agent_steps_per_s: float
    env_steps_per_s: float

    @property
    def num_agents(self) -> int:
        return self.num_envs * self.num_snakes


def _run_point(
    num_envs: int,
    num_snakes: int,
    num_steps: int,
    warmup_steps: int,
    seed: int,
    mechanics_version: int,
) -> BenchResult:
    """Time ``num_steps`` masked steps of a single (E, S) batch on CPU.

    Actions are drawn from a fixed pre-generated RNG stream so action sampling
    cost is excluded from the timed region. Masks are computed inside
    :meth:`BatchSim.step`, hence always included in the measurement.

    Args:
        num_envs: Number of parallel environments (E).
        num_snakes: Snakes per environment (S).
        num_steps: Timed frames to advance.
        warmup_steps: Untimed frames run first (populates caches / JIT-free warm).
        seed: Base seed for env RNGs and the action stream.
        mechanics_version: 1 (legacy) or 2 (trail pellets etc.).

    Returns:
        The measured :class:`BenchResult`.
    """
    cfg = BatchSimConfig(
        num_envs=num_envs,
        num_snakes=num_snakes,
        mechanics_version=mechanics_version,
    )
    sim = BatchSim(cfg, seeds=list(range(seed, seed + num_envs)), train_mode=True)
    sim.reset()

    # Pre-generate the full action tape so np.random is out of the timed loop.
    rng = np.random.default_rng(seed)
    total = warmup_steps + num_steps
    tape = rng.integers(0, 6, size=(total, num_envs, num_snakes), dtype=np.int64)

    for i in range(warmup_steps):
        sim.step(tape[i])

    t0 = time.perf_counter()
    for i in range(warmup_steps, total):
        sim.step(tape[i])
        # Touch the mask so it is materialized within the timed region.
        _ = sim.get_action_mask()
    wall = time.perf_counter() - t0

    agent_steps = float(num_envs * num_snakes * num_steps)
    env_steps = float(num_envs * num_steps)
    return BenchResult(
        num_envs=num_envs,
        num_snakes=num_snakes,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        wall_s=wall,
        agent_steps_per_s=agent_steps / wall,
        env_steps_per_s=env_steps / wall,
    )


def _format_report(results: List[BenchResult], mechanics_version: int) -> str:
    """Render measured points as a plain-text table plus gate/baseline context."""
    lines: List[str] = []
    lines.append("BatchSim throughput (CPU, single process)")
    lines.append(f"  mechanics_version = {mechanics_version}")
    lines.append("  masks: INCLUDED (computed inside BatchSim.step)")
    lines.append("  featurizer/raster: NOT included -> upper bound on dynamics half")
    lines.append("")
    header = (
        f"{'E':>5} {'S':>4} {'agents':>8} {'steps':>6} "
        f"{'wall_s':>8} {'agent_st/s':>12} {'env_st/s':>10} {'gate':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        gate = "PASS" if r.agent_steps_per_s >= GATE_AGENT_STEPS_PER_S else "FAIL"
        lines.append(
            f"{r.num_envs:>5} {r.num_snakes:>4} {r.num_agents:>8} {r.num_steps:>6} "
            f"{r.wall_s:>8.3f} {r.agent_steps_per_s:>12,.0f} "
            f"{r.env_steps_per_s:>10,.0f} {gate:>6}"
        )
    lines.append("")
    lines.append(f"  P2 gate           : >= {GATE_AGENT_STEPS_PER_S:,} agent-steps/s")
    lines.append(
        f"  audit baseline    : ~{AUDIT_ACTOR_TRANSITIONS_PER_S:,} transitions/s "
        "per single Python actor"
    )
    return "\n".join(lines)


def _default_sweep() -> List[Tuple[int, int]]:
    """(E, S) points: the headline 256x8 plus a couple of comparison shapes."""
    return [(256, 8), (64, 8), (512, 8), (256, 4), (128, 16)]


def main() -> None:
    """Parse args, run the requested benchmark point(s), print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=256, help="Environments E.")
    parser.add_argument("--snakes", type=int, default=8, help="Snakes per env S.")
    parser.add_argument("--steps", type=int, default=200, help="Timed frames.")
    parser.add_argument("--warmup", type=int, default=20, help="Untimed warmup frames.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--mechanics-version", type=int, default=2, choices=(1, 2), help="Mechanics ver."
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the headline 256x8 point plus a few comparison (E, S) shapes.",
    )
    args = parser.parse_args()

    points = _default_sweep() if args.sweep else [(args.envs, args.snakes)]
    results: List[BenchResult] = []
    for e, s in points:
        results.append(
            _run_point(
                num_envs=e,
                num_snakes=s,
                num_steps=args.steps,
                warmup_steps=args.warmup,
                seed=args.seed,
                mechanics_version=args.mechanics_version,
            )
        )

    print(_format_report(results, args.mechanics_version))


if __name__ == "__main__":
    main()
