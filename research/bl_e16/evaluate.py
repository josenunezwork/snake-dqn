"""Terminal natural-start H5000 execution with one fixed-world comparison family."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from research.bl_e16.admission import B8_SHA256, require_execution_plan
from research.bl_e16.metrics import (
    MIXES,
    decide_arm,
    holm_one_sided_bounds,
    ledger_by_mix,
    simultaneous_bonferroni_bounds,
    survival_band_v2,
    whole_life_ledger,
)
from research.bl_e16.model import E16Policy
from research.bl_info_value.reconstruct import _geometry, _history_state, _history_window
from research.bl_ppo.collector import TerminalEvalDriver, _FrozenRoster, _sim_config
from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_ppo.model import PPOPolicy
from research.published_ledger import bind_published_ledger_check, require_published_ledger
from src.model.ego2s_network import load_ego2s_checkpoint


class NaturalDriver(TerminalEvalDriver):
    """Read-only reuse of E2's selector seam, with E16's explicit mechanics v2."""

    def __init__(self, world: dict[str, Any]) -> None:
        from src.simd_env.grid_sim import GridBatchSim

        self.seed, self.horizon, self.frame = int(world["world_seed"]), int(world["horizon"]), 0
        self.sim = GridBatchSim(
            replace(_sim_config(1), mechanics_version=2),
            seeds=[self.seed],
            train_mode=False,
            allow_respawn=True,
        )
        self.frozen = _FrozenRoster(self.sim, [self.seed], [world["opponents"]])
        self.history: deque[np.ndarray] = deque(maxlen=128)

    def observe_hero(self) -> dict[str, np.ndarray]:
        obs = super().observe_hero()
        obs["geometry"] = _geometry(self.sim)[None]
        obs["history"] = _history_window(self.history)[None]
        obs["frame"] = np.array([self.frame], dtype=np.int64)
        return obs

    def step(self, action: int) -> dict[str, Any]:
        if self.frame >= self.horizon or not self.sim.get_alive()[0, 0]:
            raise RuntimeError("terminal driver is finished")
        actions = np.ones((1, self.sim.S), dtype=np.int64)
        actions[0, 0] = int(action)
        self.frozen.actions(self.sim, actions)
        self.history.append(_history_state(self.sim, actions))
        self.sim.step(actions)
        self.frozen.observe_step(self.sim)
        self.frame += 1
        alive = bool(self.sim.get_alive()[0, 0])
        return {
            "done": not alive or self.frame >= self.horizon,
            "alive": alive,
            "mass": float(self.sim.length[0, 0]) if alive else 0.0,
            "frame": self.frame,
        }


def load_controller(name: str, checkpoint: Path) -> Any:
    """Load immutable B8, nominated F, or a portable final E16 actor."""
    if name == "B8":
        return load_ego2s_checkpoint(str(checkpoint), "cpu").eval()
    verify_export(checkpoint)
    if name == "parent":
        return PPOPolicy.from_checkpoint(checkpoint, "cpu").eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = E16Policy.from_portable_state(payload.get("policy", payload)).eval()
    if policy.arm != name:
        raise ValueError("endpoint arm identity differs")
    return policy


def evaluate_controller(
    world: dict[str, Any], name: str, model: Any, driver_factory: Callable[..., Any] = NaturalDriver
) -> dict[str, Any]:
    """Closed loop for a whole life; no candidate handback, intervention or respawn."""
    horizon = int(world["horizon"])
    driver = driver_factory(world)
    lengths, trace = [0.0] * horizon, hashlib.sha256()
    started = time.monotonic()
    boost_count = 0
    initial_length = float(driver.sim.length[0, 0]) if hasattr(driver, "sim") else None
    for frame in range(horizon):
        obs = driver.observe_hero()
        tensors = {key: torch.as_tensor(value) for key, value in obs.items()}
        with torch.inference_mode():
            args = [tensors[k] for k in ("local", "global", "scalars")]
            if name == "B8":
                logits = model(*args)
            elif name == "parent":
                logits, _, _ = model(*args)
            else:
                logits = model.actor_logits(
                    *args, geometry=tensors.get("geometry"), history=tensors.get("history")
                )
        values = logits.detach().cpu().numpy().reshape(6)
        if not np.isfinite(values).all():
            raise ValueError("non-finite endpoint logits")
        mask = np.asarray(obs["mask"], dtype=bool).reshape(6)
        if not mask.any():
            raise ValueError("no legal endpoint action")
        action = int(np.where(mask, values.astype(np.float64), -np.inf).argmax())
        for key in sorted(obs):
            array = np.ascontiguousarray(obs[key])
            trace.update(str((key, array.shape, array.dtype.str)).encode())
            trace.update(array.tobytes())
        trace.update(values.tobytes())
        boost_count += int(action >= 3)
        step = driver.step(action)
        trace.update(bytes([action, bool(step["alive"])]))
        lengths[frame] = float(step["mass"]) if step["alive"] else 0.0
        if step["done"]:
            if step["alive"] and frame != horizon - 1:
                raise ValueError("early truncation cannot produce whole-life evidence")
            break
    return {
        "boost_count": boost_count,
        "lengths": lengths,
        "initial_length": initial_length,
        "trace_sha256": trace.hexdigest(),
        "active_frames": frame + 1,
        "mass_integral": sum(lengths) / horizon,
        "alive_fraction": sum(x > 0 for x in lengths) / horizon,
        "wall_seconds": time.monotonic() - started,
    }


def _world_task(task: tuple[dict[str, Any], dict[str, str]]) -> dict[str, Any]:
    torch.set_num_threads(1)
    world, checkpoints = task
    controllers = {
        name: evaluate_controller(world, name, load_controller(name, Path(path)))
        for name, path in checkpoints.items()
    }
    return {
        "world_id": world["world_id"],
        "world_seed": world["world_seed"],
        "mix": world["mix"],
        "horizon": world["horizon"],
        "controllers": controllers,
    }


def execute_worlds(
    worlds: list[dict[str, Any]], checkpoints: dict[str, Path], workers: int = 1
) -> list[dict[str, Any]]:
    """Each worker owns ALL controllers of a pair; map preserves frozen bank order."""
    tasks = [(world, {key: str(value) for key, value in checkpoints.items()}) for world in worlds]
    if workers == 1:
        return [_world_task(task) for task in tasks]
    if not 1 <= workers <= 12:
        raise ValueError("workers must be 1..12")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(_world_task, tasks))


def analyze(
    rows: list[dict[str, Any]],
    arms: list[str],
    *,
    scientific: bool,
    expected_worlds: list[dict[str, Any]],
    training_status: dict[str, Any],
) -> dict[str, Any]:
    """Analyze only a complete frozen family, using paired-world simultaneous bounds."""
    expected = {str(row["world_id"]): row for row in expected_worlds}
    if len(rows) != len(expected) or len({r["world_id"] for r in rows}) != len(rows):
        raise ValueError("missing or duplicate paired worlds")
    expected_controllers = {"B8", "parent", *arms}
    for row in rows:
        world = expected.get(row["world_id"])
        if world is None or any(row[k] != world[k] for k in ("world_seed", "mix", "horizon")):
            raise ValueError("world identity differs from frozen bank")
        if set(row["controllers"]) != expected_controllers or (
            scientific and row["horizon"] != 5000
        ):
            raise ValueError("controller family or whole-life horizon differs")
        for controller in row["controllers"].values():
            values = controller["lengths"]
            whole_life_ledger(values, values, horizon=row["horizon"])
            expected_alive = sum(value > 0 for value in values) / row["horizon"]
            if (
                not np.isfinite(controller["alive_fraction"])
                or abs(controller["alive_fraction"] - expected_alive) > 1e-12
            ):
                raise ValueError("alive-frame counter reconciliation failed")
            if not np.isfinite(controller["mass_integral"]):
                raise ValueError("nonfinite mass integral")
            if abs(controller["mass_integral"] - sum(values) / row["horizon"]) > 1e-9:
                raise ValueError("mass counter reconciliation failed")
    if scientific and any(sum(r["mix"] == m for r in rows) != 96 for m in MIXES):
        raise ValueError("fixed 96 worlds/mix cannot be shrunk")
    if not scientific:
        return {
            "schema": "bl-e16-evaluation/v1",
            "status": "SMOKE_ONLY",
            "scientific": False,
            "worlds": len(rows),
            "rankings": {arm: {"status": "UNRANKED"} for arm in arms},
        }
    groups, ledgers, claims = {}, {}, {}
    for arm in arms:
        groups[arm] = {metric: {mix: [] for mix in MIXES} for metric in ("J", "G", "collateral")}
        ledger_rows = []
        for row in rows:
            c = row["controllers"]
            item = whole_life_ledger(c[arm]["lengths"], c["B8"]["lengths"])
            for metric, value in (
                ("J", c[arm]["mass_integral"] - c["parent"]["mass_integral"]),
                ("G", item["total_delta"]),
                ("collateral", item["collateral"]),
            ):
                groups[arm][metric][row["mix"]].append(value)
            ledger_rows.append(
                {
                    "mix": row["mix"],
                    "candidate_lengths": c[arm]["lengths"],
                    "b8_lengths": c["B8"]["lengths"],
                }
            )
        ledgers[arm] = ledger_by_mix(ledger_rows)
        for metric in ("J", "G", "collateral"):
            claims[f"{arm}:{metric}"] = {
                "groups": groups[arm][metric],
                "direction": "lower" if metric == "J" else "upper",
            }
    simultaneous = simultaneous_bonferroni_bounds(claims)
    holm = holm_one_sided_bounds({arm: groups[arm]["J"] for arm in arms})
    decisions = {}
    for arm in arms:
        j = {
            **simultaneous[f"{arm}:J"],
            "holm_rejects_nonpositive": holm[arm]["holm_rejects_nonpositive"],
        }
        survival = survival_band_v2(
            {
                m: [r["controllers"][arm]["alive_fraction"] for r in rows if r["mix"] == m]
                for m in MIXES
            },
            {
                m: [r["controllers"]["B8"]["alive_fraction"] for r in rows if r["mix"] == m]
                for m in MIXES
            },
        )
        status = training_status.get(arm, {})
        valid = (
            status.get("status") == "COMPLETE"
            and status.get("critic_qualified") is True
            and status.get("critic_transitions") == 2_000_000
            and status.get("actor_transitions") == 4_000_000
        )
        if status.get("status") == "KILL" and status.get("tripwires"):
            decisions[arm] = {
                "status": "KILL",
                "reasons": status["tripwires"],
                "efficacy": "NOT_INTERPRETED",
            }
            continue
        decisions[arm] = {
            **decide_arm(
                available=True,
                valid=valid,
                tripwires=status.get("tripwires", []),
                j=j,
                g=simultaneous[f"{arm}:G"],
                ledger=ledgers[arm],
                collateral=simultaneous[f"{arm}:collateral"],
                survival=survival,
            ),
            "J": j,
            "G": simultaneous[f"{arm}:G"],
            "ledger": ledgers[arm],
            "collateral": simultaneous[f"{arm}:collateral"],
            "survival_guard_A": survival,
        }
    for arm in ("R", "S", "P"):
        decisions[arm] = {"status": "UNRANKED", "reason": "gated implementation stub"}
    return {
        "schema": "bl-e16-evaluation/v1",
        "status": "COMPLETE",
        "scientific": True,
        "family_alpha": 0.10,
        "look": 1,
        "arms": decisions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--endpoint-lock", type=Path, required=True)
    parser.add_argument("--authoring-repo", type=Path, required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args(argv)
    proof = require_published_ledger(args.authoring_repo, args.ledger)
    plan = require_execution_plan(args.plan)
    verify_export(args.endpoint_lock)
    lock = json.loads(args.endpoint_lock.read_text())
    if (
        lock.get("schema") != "bl-e16-endpoint-lock/v1"
        or lock.get("plan_sha256") != digest(args.plan)
        or lock.get("efficacy_unread") is not True
        or lock.get("arms") != plan["arms"]
    ):
        raise ValueError("final endpoints must be frozen before the single efficacy look")
    plan = {**plan, "endpoints": lock["endpoints"], "training_status": lock["training_status"]}
    base = Path(plan["_base"])
    checkpoints = {name: base / binding["path"] for name, binding in plan["endpoints"].items()}
    if set(checkpoints) != {"B8", "parent", *plan["arms"]}:
        raise ValueError("all frozen endpoints required")
    if plan["endpoints"]["parent"]["sha256"] != plan["parent"]["sha256"]:
        raise ValueError("J comparator differs from nominated common parent")
    if plan["endpoints"]["B8"]["sha256"] != B8_SHA256:
        raise ValueError("G comparator differs from released B8")
    for name, checkpoint in checkpoints.items():
        if digest(checkpoint) != plan["endpoints"][name]["sha256"]:
            raise ValueError("endpoint SHA mismatch")
    statuses = {
        arm: json.loads((base / plan["training_status"][arm]["path"]).read_text())
        for arm in plan["arms"]
    }
    for arm in plan["arms"]:
        if (
            verify_export(base / plan["training_status"][arm]["path"])
            != plan["training_status"][arm]["sha256"]
        ):
            raise ValueError("training status SHA mismatch")
    for arm in plan["arms"]:
        record = statuses[arm]
        identity = record.get("identity", {})
        if (
            record.get("endpoint_sha256") != plan["endpoints"][arm]["sha256"]
            or identity.get("parent_sha256") != plan["parent"]["sha256"]
            or identity.get("bank_sha256") != plan["bank"]["sha256"]
            or identity.get("b8_sha256") != B8_SHA256
            or identity.get("plan_sha256") != digest(args.plan)
            or identity.get("scientific") is not True
        ):
            raise ValueError("training status does not bind the frozen endpoint and family")
    rows = execute_worlds(plan["_bank"]["rows"], checkpoints, args.workers)
    result = analyze(
        rows,
        plan["arms"],
        scientific=True,
        expected_worlds=plan["_bank"]["rows"],
        training_status=statuses,
    )
    export_json(args.output.with_name(args.output.stem + "-worlds.json"), {"rows": rows})
    export_json(args.output, bind_published_ledger_check(result, proof))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
