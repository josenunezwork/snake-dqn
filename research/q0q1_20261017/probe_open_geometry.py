#!/usr/bin/env python3
"""Descriptive, bounded geometry probe for Q0's fixed B8 audit worlds.

This is deliberately separate from the fidelity receipt.  It runs three short,
real evaluation episodes and reports the population and nearest-other-cell
geometry that motivated the local-open operational definition.  It does not
train, retain fidelity candidates, or decide any efficacy endpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.q0q1_20261017.audit_common import B8_SHA256, sha256_file, write_json  # noqa: E402
from research.q0q1_20261017.audit_episode_metrics import locally_open  # noqa: E402

SCHEMA = "q0-open-geometry-probe/v1"
DEFAULT_HORIZON = 300
DEFAULT_SEED = 7101
MIXES = ("scripted", "frozen", "mixed")


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return float(ordered[round((len(ordered) - 1) * fraction)])

    return {
        "min": float(ordered[0]),
        "p50": at(0.5),
        "p95": at(0.95),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


class GeometryObserver:
    """Read-only hook around the hero policy's real pre-action simulator state."""

    def __init__(self) -> None:
        self.alive_counts: list[int] = []
        self.chebyshev: list[float] = []
        self.euclidean: list[float] = []
        self.local_open: list[bool] = []

    def observe(self, sim: Any, slots: np.ndarray) -> None:
        owners, _, _ = sim.get_grids()
        heads = sim.get_heads()
        alive = sim.get_alive()
        for env, hero in np.asarray(slots, dtype=np.int64).reshape(-1, 2):
            if int(hero) != 0 or not bool(alive[env, hero]):
                continue
            hx, hy = (int(value) for value in heads[env, hero])
            cells = np.argwhere((owners[env] >= 0) & (owners[env] != hero))
            # Interior grid coordinates are (row=y, col=x), while heads are (x, y).
            if not len(cells):
                continue
            dx = np.abs(cells[:, 1] - hx)
            dy = np.abs(cells[:, 0] - hy)
            self.alive_counts.append(int(alive[env].sum()))
            self.chebyshev.append(float(np.minimum.reduce(np.maximum(dx, dy))))
            self.euclidean.append(float(np.sqrt(dx * dx + dy * dy).min()))
            self.local_open.append(locally_open(sim, int(env), int(hero)))

    def result(self) -> dict[str, Any]:
        return {
            "hero_decisions_observed": len(self.alive_counts),
            "alive_snakes": {
                "counts": {
                    str(key): value for key, value in sorted(Counter(self.alive_counts).items())
                },
                "summary": _quantiles([float(value) for value in self.alive_counts]),
            },
            "nearest_other_snake_cell": {
                "chebyshev_cells": _quantiles(self.chebyshev),
                "euclidean_cells": _quantiles(self.euclidean),
            },
            "locally_open": {
                "true": int(sum(self.local_open)),
                "false": int(len(self.local_open) - sum(self.local_open)),
            },
        }


@contextmanager
def observe_hero_actions(observer: GeometryObserver) -> Iterator[None]:
    """Patch only the evaluator's hero policy call; actions remain unmodified."""
    from unittest.mock import patch

    from src.simd_env.ego2s_policy import Ego2sSimdPolicy

    original = Ego2sSimdPolicy.actions

    def observed(policy, masks, sim, slots):
        observer.observe(sim, slots)
        return original(policy, masks, sim, slots)

    with patch.object(Ego2sSimdPolicy, "actions", observed):
        yield


def _worlds(seed: int) -> list[Mapping[str, Any]]:
    from research.q0q1_20261017 import audit_q0_bank

    bank = audit_q0_bank.build_world_bank()
    rows = list(bank["worlds"][str(seed)])
    return [next(row for row in rows if row["mix"] == mix) for mix in MIXES]


def run(b8: Path, *, seed: int = DEFAULT_SEED, horizon: int = DEFAULT_HORIZON) -> dict[str, Any]:
    """Run one fixed Q0 world per mix for a short, descriptive horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not b8.is_file() or sha256_file(b8) != B8_SHA256:
        raise ValueError("geometry probe requires released B8 bytes")
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t23_r0_screen_20261017 import evaluate as t23_eval
    from src.simd_env.eval_engine import run_simd_eval

    context = gid._context(1)
    # Preserve the same mechanics, roster, masks, and initialization world while
    # explicitly making this a short descriptive prefix rather than an H5000 gate.
    profile = replace(context["profile"], scored_horizon=horizon)
    started = time.monotonic()
    episodes = []
    for number, world in enumerate(_worlds(seed), 1):
        roster = t23_eval._opponents(world, context["lookup"])
        observer = GeometryObserver()
        with observe_hero_actions(observer):
            run_simd_eval(
                context["lookup"][gid.FRP3_S12[1]],
                roster,
                horizon,
                [int(world["world_seed"])],
                profile=profile,
                opponent_specs_by_world={int(world["world_seed"]): roster},
                mix_id=str(world["mix"]),
                vector61=True,
                vector61_forward="rowwise",
                sim_engine="grid",
                hero_ego2s=str(b8),
                ego2s_forward="rowwise",
                ego2s_mask="advisory",
                vector61_backend="numba",
                scripted_anchors="fast",
            )
        result = observer.result()
        episodes.append(
            {
                "mix": str(world["mix"]),
                "world_seed": int(world["world_seed"]),
                "world_semantic_sha256": str(world["world_semantic_sha256"]),
                **result,
            }
        )
        print(
            f"Q0 geometry probe episodes {number}/{len(MIXES)} mix={world['mix']} done", flush=True
        )
    return {
        "schema": SCHEMA,
        "authority": "descriptive geometry only; no fidelity or efficacy decision",
        "b8_sha256": sha256_file(b8),
        "seed": seed,
        "horizon": horizon,
        "geometry": {
            "local_raster": "31x31, hero row=23 col=15; ahead=[-7,23], lateral=[-15,15] inclusive",
            "open": "no other living-snake-owned cell in the actual heading-rotated local crop",
        },
        "episodes": episodes,
        "wall_seconds": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args(argv)
    report = run(args.b8, seed=args.seed, horizon=args.horizon)
    write_json(args.out, report)
    print(f"Q0 geometry probe {SCHEMA} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
