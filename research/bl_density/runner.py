"""Development-only density measurement collector.

This deliberately reuses the Watch selector seam used by the dense-search
fixtures.  It is neither a gate nor a source of scientific conclusions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from research.bl_ppo.exports import export_json

NAMESPACE = "bl-density-round14-20261022"
HORIZON, DEPTH = 5000, 300
MIXES = ("scripted", "frozen", "mixed")
ARMS = ("A", "B70m", "B70p", "C-P1")
FEATURE_NAMES = tuple(
    f"{action}_{name}"
    for action in range(6)
    for name in ("area_g", "head_risky", "landing_ok", "v5_eligible", "spacious")
) + ("q_gap", "q_max", "q_min", "q_mean", "q_std")


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def cheap_prefilter(counts: np.ndarray, advisory: np.ndarray) -> bool:
    """Frozen broad screen used only to avoid unnecessary H300 continuations."""
    values = np.asarray(counts, dtype=np.int64).reshape(-1)
    mask = np.asarray(advisory, dtype=bool).reshape(-1)
    if values.shape != (3,) or mask.shape != (6,):
        raise ValueError("malformed free-space counts or advisory mask")
    return bool(values.min() < 160 or mask[:3].sum() <= 2)


def effective_q_features(q: np.ndarray, advisory: np.ndarray) -> tuple[int, list[float]]:
    """Return B8's fallback action and the frozen five Q summaries."""
    q = np.asarray(q, dtype=np.float64)
    advisory = np.asarray(advisory, dtype=bool)
    if q.shape != (6,) or advisory.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("malformed Q or advisory mask")
    allowed = advisory.copy()
    if not allowed.any():
        allowed[:3] = True
    values = q[allowed]
    ordered = np.sort(values)[::-1]
    gap = float(ordered[0] - ordered[1]) if ordered.size > 1 else 0.0
    action = int(np.where(allowed, q, -np.inf).argmax())
    return action, [
        gap,
        float(values.max()),
        float(values.min()),
        float(values.mean()),
        float(values.std()),
    ]


def legal_alternatives(legal_mask: np.ndarray, bare_action: int) -> tuple[int, ...]:
    """Return every physically legal counterfactual, including advisory-rejected moves."""
    mask = np.asarray(legal_mask, dtype=bool)
    if mask.shape != (6,) or not 0 <= int(bare_action) < 6:
        raise ValueError("malformed legal action mask or bare action")
    return tuple(int(action) for action in np.flatnonzero(mask) if int(action) != int(bare_action))


def retrospective_doom(death_decision: int | None, decision: int) -> bool:
    """Whether a natural fatal transition lies inside the contract's H300 window."""
    return death_decision is not None and 0 <= int(death_decision) - int(decision) < DEPTH


def _seed(arm: str, mix: str, index: int) -> int:
    raw = f"{NAMESPACE}/{arm}/{mix}/{index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _expanded_specs(specs: Sequence[Sequence[str]], count: int) -> list[list[str]]:
    return [list(specs[index % len(specs)]) for index in range(count)]


@contextlib.contextmanager
def _profile_override(*, area: float | None, snakes: int) -> Iterator[None]:
    """Temporarily retain the standard profile knobs while changing only density."""
    from src.simd_env import eval_engine

    original = eval_engine._config_from_game_config

    def replacement(num_snakes: int, gamma: float, profile: Any = None) -> Any:
        base_count = int(profile.world.num_snakes) if profile is not None else num_snakes
        cfg = original(base_count, gamma, profile)
        if area is None:
            return dataclasses.replace(cfg, num_snakes=snakes)
        factor = float(area) ** 0.5
        cell = int(cfg.segment_size)
        width = max(cell, int(cfg.game_width * factor) // cell * cell)
        height = max(cell, int(cfg.game_height * factor) // cell * cell)
        return dataclasses.replace(cfg, num_snakes=snakes, game_width=width, game_height=height)

    eval_engine._config_from_game_config = replacement
    try:
        yield
    finally:
        eval_engine._config_from_game_config = original


def _live_body_cells(sim: Any, slot: int) -> list[np.ndarray]:
    """Read exactly the live ring cells for one snake, head first."""
    if not bool(sim.alive[0, slot]):
        return []
    count = int(sim.seg_count[0, slot])
    head = int(sim.head_ptr[0, slot])
    indices = (head - np.arange(count, dtype=np.int64)) % int(sim.cap)
    return [np.asarray(sim.bodies[0, slot, index], dtype=np.int64) for index in indices]


def _geometry_distances(sim: Any) -> dict[str, int | None]:
    hero_cells = _live_body_cells(sim, 0)
    if not hero_cells:
        raise RuntimeError("cannot describe a sampled root with a dead hero")
    hero = hero_cells[0]
    opponents = [_live_body_cells(sim, slot) for slot in range(1, sim.S)]
    heads = [body[0] for body in opponents if body]
    head_distance = min((int(np.abs(hero - other).sum()) for other in heads), default=None)
    body_cells = [cell for body in opponents for cell in body[1:]]
    body_distance = min((int(np.abs(hero - cell).sum()) for cell in body_cells), default=None)
    return {
        "nearest_opponent_head_manhattan": head_distance,
        "nearest_opponent_body_manhattan": body_distance,
    }


def _unique_occupied_cells(sim: Any) -> int:
    """Count physical occupancy, retaining the amendment's U_total definition."""
    cells: set[tuple[int, int]] = set()
    for slot in range(sim.S):
        for cell in _live_body_cells(sim, slot):
            cells.add((int(cell[0]), int(cell[1])))
    return len(cells)


def _death_cause_label(sim: Any) -> str | None:
    """Return the simulator's terminal cause for the hero's last transition."""
    from src.simd_env import DEATH_BODY, DEATH_HEAD, DEATH_NONE, DEATH_SELF, DEATH_WALL

    labels = {
        DEATH_NONE: None,
        DEATH_WALL: "DEATH_WALL",
        DEATH_SELF: "DEATH_SELF",
        DEATH_HEAD: "DEATH_HEAD",
        DEATH_BODY: "DEATH_BODY",
    }
    cause = int(sim.get_death_cause()[0, 0])
    if cause not in labels:
        raise RuntimeError(f"unknown GridBatchSim death cause {cause}")
    return labels[cause]


def _root_geometry(sim: Any, arm: str) -> dict[str, Any]:
    """Capture the fixed root geometry and the prospective minimum-map diagnostic."""
    from src.simd_env.batch_sim import CARDINAL

    lengths = [int(value) for value in sim.length[0].tolist()]
    occupied = _unique_occupied_cells(sim)
    grid_cells = int(sim.grid_w * sim.grid_h)
    hero = sim.heads()[0, 0]
    dx, dy = (int(value) for value in CARDINAL[int(sim.direction[0, 0])])
    x, y = (int(value) for value in hero)
    wall_ahead = (
        x if dx < 0 else sim.grid_w - 1 - x if dx > 0 else y if dy < 0 else sim.grid_h - 1 - y
    )
    hero_length = lengths[0]
    known_floor = max(8 * hero_length, 8349)
    # rho95 is deliberately unavailable until arm A's prospective reference is
    # collected.  A shrunken arm is consequently diagnostic, never certified
    # transferable, even where it satisfies the known two-term floor.
    minimum_map = {
        "formula": "max(8*L_hero,U_total/rho95,0.7*A0)",
        "a0_cells": 145 * 83,
        "rho95": None,
        "rho95_reference_status": "PENDING_ARM_A_ROOTS",
        "known_floor_cells": known_floor,
        "map_cells": grid_cells,
        "known_terms_satisfied": grid_cells >= known_floor,
        "transferability_status": (
            "STANDARD_MAP_REFERENCE" if arm in {"A", "C-P1"} else "DIAGNOSTIC_RHO95_UNMEASURED"
        ),
    }
    assessment = {
        **minimum_map,
        "source_eligible": grid_cells >= known_floor,
        "source_eligibility_reason": (
            "KNOWN_MINIMUM_MAP_VIOLATION" if grid_cells < known_floor else "RHO95_PENDING"
        ),
    }
    return {
        "all_snake_lengths": lengths,
        "all_snake_alive": [bool(value) for value in sim.alive[0].tolist()],
        "total_occupied_cells": occupied,
        "map_cells": grid_cells,
        "total_occupied_fraction": occupied / grid_cells,
        "hero_occupancy": hero_length / grid_cells,
        "wall_distance_ahead": int(wall_ahead),
        **_geometry_distances(sim),
        "minimum_map_rule": minimum_map,
        "minimum_map_assessment": assessment,
    }


def _inject_p1_long(sim: Any, hero_length: int) -> None:
    """Install P1's registered long-body fixture using its production geometry helper."""
    if hero_length not in (900, 1200, 1500):
        raise ValueError("C-P1 hero length must be 900, 1200, or 1500")
    if hero_length == 900:
        # Use P1's helper directly where its registered tuple is exact.
        from research.bl_ppo.collector import _inject_long

        _inject_long(sim)
        return
    from src.simd_env.grid_scenarios import inject_serpentines

    # P1's helper delegates to this same primitive; only the registered hero
    # stratum changes while all five opponent lengths remain byte-for-byte equal.
    # A band must contain the hero's whole boustrophedon path: 1200 and 1500
    # require 9 and 11 rows on the 145-cell standard width, respectively.
    box_height = max(8, math.ceil(hero_length / (int(sim.grid_w) - 4)))
    inject_serpentines([sim], (hero_length, 600, 400, 300, 150, 150), box_height=box_height)


def _native_hook_state(world: Any) -> dict[str, str | None]:
    """Disclose the actual evaluator hook path without claiming a required S2 hook."""
    opponents = getattr(world, "opponents", ())
    values = [getattr(policy, "native_v8_actual", None) for policy in opponents]
    return {
        "runtime_hook": f"{type(world.runtime).__name__}.prepare/observe_step",
        "s2_native_v8": next((str(value) for value in values if value is not None), None),
    }


def _rollout_continuation(
    world: Any, action: int, bare_action: int, decision: int
) -> dict[str, Any]:
    """Run one exact forced-first-action continuation from a prepared selector root.

    This is local rather than ``rollout_all`` because the frozen shared helper does
    not export death cause.  Each clone starts at the evaluator's prepared selector
    seam, so opponent carry, RNG, and runtime prepare/observe hooks remain intact.
    """
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step, step_prepared

    started = time.process_time()
    branch = world.clone()
    # Match the shared continuation seam: shallow policy copies must not share
    # mutable diagnostics between the original world and counterfactual siblings.
    for policy in [branch.hero, *branch.opponents]:
        for name in ("live_vetoes", "veto_counters", "boost_counters", "last_veto"):
            if hasattr(policy, name):
                setattr(policy, name, copy.deepcopy(getattr(policy, name)))
    executed = 0
    step_prepared(branch, int(action))
    executed += 1
    while executed < DEPTH and bool(branch.sim.alive[0, 0]):
        choose_and_step(branch, capture_obs=False)
        executed += 1
    died = not bool(branch.sim.alive[0, 0])
    return {
        "action": int(action),
        "is_bare_action": int(action) == bare_action,
        "survives": not died and executed == DEPTH,
        "death_decision": int(decision) + executed - 1 if died else None,
        "death_cause": _death_cause_label(branch.sim) if died else None,
        "frames_executed": executed,
        "cpu_seconds": time.process_time() - started,
    }


def _build(
    plan: dict[str, Any], b8: Path, arm: str, target: int | None
) -> tuple[Any, dict[str, Any]]:
    from research.t3_rollout_labels_20261015.bank_engine import build_world

    specs = list(plan["specs"])
    area = 0.7 if arm in {"B70m", "B70p"} else None
    if arm == "B70p":
        plan = {**plan, "specs": _expanded_specs(specs, 7)}
    with _profile_override(area=area, snakes=8 if arm == "B70p" else 6):
        world = build_world("b8_s3", plan, hero_checkpoint=b8)
    world._density_arm = arm
    details: dict[str, Any] = {
        "sim_config": {
            key: getattr(world.sim.cfg, key)
            for key in (
                "game_width",
                "game_height",
                "segment_size",
                "num_snakes",
                "initial_food",
                "max_food",
            )
        },
        "roster_specs": plan["specs"],
    }
    cfg = details["sim_config"]
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    standard = gid._context(1)["profile"].world
    details["realized_density"] = {
        "area_fraction": (cfg["game_width"] * cfg["game_height"])
        / (standard.width * standard.height),
        "snakes_per_map_cell": cfg["num_snakes"]
        / (
            (cfg["game_width"] // cfg["segment_size"]) * (cfg["game_height"] // cfg["segment_size"])
        ),
    }
    if arm == "C-P1":
        _inject_p1_long(world.sim, int(target))
        details["fixture"] = {
            "construction": "bl_ppo.collector._inject_long/inject_serpentines",
            "body_lengths": [int(target), 600, 400, 300, 150, 150],
            "synthetic": True,
        }
        details["fixture"].update(_geometry_distances(world.sim))
    return world, details


def _step(
    world: Any,
    decision: int,
    horizon: int,
    rows: list[dict[str, Any]],
    lengths: list[int],
    h300: list[dict[str, Any]],
    stages: dict[str, float],
    confirmed_dooms: list[int],
) -> bool:
    """One exact prepare/observe/dispatch transition, optionally screening a root."""
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.eval_engine import _dispatch_actions

    selected: dict[str, Any] = {}

    def choose(prepared: Any) -> np.ndarray:
        masks = prepared.get_resolved_action_mask()
        world.runtime.prepare(prepared)
        alive = bool(prepared.alive[0, 0])
        length = int(prepared.length[0, 0]) if alive else 0
        if alive:
            lengths.append(length)
        advisory = prepared.get_advisory_action_mask()[0, 0].astype(bool)
        obs = build_ego_raster(
            prepared, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
        )
        q = np.asarray(prepared_q(world, obs), dtype=np.float32)
        _, q_stats = effective_q_features(q, advisory)
        actions = np.ones((1, prepared.S), dtype=np.int64)
        _dispatch_actions(prepared, masks, actions, [world.hero], [world.opponents])
        bare = int(actions[0, 0])
        stride = 50 if horizon < HORIZON else 200
        sampled = (
            alive
            and length >= 800
            and decision % stride == 0
            and decision <= min(4700, horizon - 1)
        )
        record: dict[str, Any] | None = None
        if sampled:
            screen_started = time.process_time()
            counts = world.runtime.decision_free_space(
                prepared, np.array([[0, 0]], dtype=np.int64)
            )[0]
            geometry = _root_geometry(prepared, str(world._density_arm))
            stages["screen"] += time.process_time() - screen_started
            source_eligible = bool(geometry["minimum_map_assessment"]["source_eligible"])
            cheap_passed = cheap_prefilter(counts, advisory)
            passed = cheap_passed
            record = {
                "decision": decision,
                "length": length,
                "prefilter": passed,
                "cheap_prefilter_passed": cheap_passed,
                "source_eligible": source_eligible,
                "free_space_counts": [int(x) for x in counts],
                **geometry,
            }
            if passed:
                legal = tuple(
                    int(action) for action in np.flatnonzero(prepared.get_legal_action_mask()[0, 0])
                )
                alternatives = legal_alternatives(prepared.get_legal_action_mask()[0, 0], bare)
                outcome = _rollout_continuation(world, bare, bare, decision)
                doom = not bool(outcome["survives"])
                root_cpu = float(outcome["cpu_seconds"])
                continuations = [outcome]
                alternatives_cpu = 0.0
                alternatives_assessed = doom and confirmed_dooms[0] < 4 and bool(alternatives)
                if doom:
                    confirmed_dooms[0] += 1
                if alternatives_assessed:
                    for action in alternatives:
                        alternative = _rollout_continuation(world, action, bare, decision)
                        continuations.append(alternative)
                        alternatives_cpu += float(alternative["cpu_seconds"])
                stages["root"] += root_cpu
                stages["alternatives"] += alternatives_cpu
                record.update(
                    h300_doom=doom,
                    h300_death_decision=outcome.get("death_decision"),
                    death_cause=outcome["death_cause"],
                    h300_cpu_seconds=root_cpu + alternatives_cpu,
                    h300_frames_executed=int(outcome.get("frames_executed", DEPTH)),
                    continuation_branch_decisions=sum(
                        int(value["frames_executed"]) for value in continuations
                    ),
                    features=[float(x) for x in obs["scalars"][0, -30:].tolist() + q_stats],
                    all_legal_actions=list(legal),
                    bare_action=bare,
                    alternative_count=len(alternatives),
                    alternatives_assessed=alternatives_assessed,
                    continuations=continuations,
                    witnessed_avoidable=(
                        doom
                        and any(
                            bool(value["survives"])
                            for value in continuations
                            if not value["is_bare_action"]
                        )
                        if alternatives_assessed
                        else None
                    ),
                )
                h300.append(record)
            else:
                record["witnessed_avoidable"] = None
            rows.append(record)
        selected["alive"] = alive
        return actions

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    return bool(selected.get("alive", False))


def run_episode(job: dict[str, Any]) -> dict[str, Any]:
    """Run one deterministic arm/mix episode and preserve every sampled root."""
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    b8, arm, mix = Path(job["b8"]), str(job["arm"]), str(job["mix"])
    horizon = int(job["horizon"])
    target = (900, 1200, 1500)[int(job["index"]) % 3] if arm == "C-P1" else None
    plan = dict(job["plan"])
    world, construction = _build(plan, b8, arm, target)
    setup_cpu = time.process_time() - start_cpu
    stages = {
        "setup": setup_cpu,
        "prefix": 0.0,
        "screen": 0.0,
        "root": 0.0,
        "alternatives": 0.0,
    }
    sampled: list[dict[str, Any]] = []
    h300: list[dict[str, Any]] = []
    confirmed_dooms = [0]
    lengths: list[int] = []
    decision = 0
    step_cpu: list[float] = []
    deadline = job.get("deadline")
    while decision < horizon and bool(world.sim.alive[0, 0]):
        if deadline is not None and time.monotonic() >= float(deadline):
            break
        tick = time.process_time()
        _step(world, decision, horizon, sampled, lengths, h300, stages, confirmed_dooms)
        step_cpu.append(time.process_time() - tick)
        decision += 1
    death = decision - 1 if not bool(world.sim.alive[0, 0]) else None
    for row in sampled:
        # A tiny horizon and a cooperative deadline have the same evidence
        # boundary: an alive original trajectory cannot label an unobserved
        # 300-transition suffix as non-doom.
        unknown = death is None and decision - int(row["decision"]) < DEPTH
        row["natural_doom"] = None if unknown else retrospective_doom(death, row["decision"])
        if (
            row["prefilter"]
            and row["natural_doom"] is not None
            and bool(row["h300_doom"]) != bool(row["natural_doom"])
        ):
            raise RuntimeError("H300 continuation disagrees with uninterrupted natural trajectory")
    cpu_seconds = time.process_time() - start_cpu
    branch_cpu = sum(float(row.get("h300_cpu_seconds", 0.0)) for row in h300)
    complete = decision >= horizon or death is not None
    live_cpu = cpu_seconds - branch_cpu - setup_cpu
    later_branch_cpu = sum(
        float(row.get("h300_cpu_seconds", 0.0)) for row in h300 if row["decision"] > 0
    )
    first_branch_cpu = sum(
        float(row.get("h300_cpu_seconds", 0.0)) for row in h300 if row["decision"] == 0
    )
    steady = sum(step_cpu[1:]) - later_branch_cpu
    stages["prefix"] = max(
        0.0,
        cpu_seconds - stages["setup"] - stages["screen"] - stages["root"] - stages["alternatives"],
    )
    return {
        "schema": "bl-density-episode/v2",
        "episode_id": f"{NAMESPACE}/{arm}/{mix}/{job['index']:03d}",
        "namespace": NAMESPACE,
        "arm": arm,
        "mix": mix,
        "seed": plan["seed"],
        "plan": plan,
        "sample_stride": 50 if horizon < HORIZON else 200,
        "horizon": horizon,
        "complete": complete,
        "decisions": decision,
        "hero_decisions": decision,
        "death_decision": death,
        "death_cause": _death_cause_label(world.sim) if death is not None else None,
        "lengths": lengths,
        "sampled": sampled,
        "h300_calls": len(h300),
        "h300_branch_decisions": sum(
            int(row.get("continuation_branch_decisions", 0)) for row in h300
        ),
        "h300_cpu_seconds": branch_cpu,
        "cpu_seconds_by_stage": stages,
        "native_hook_state": _native_hook_state(world),
        "world_setup_cpu_seconds": setup_cpu,
        "live_collection_cpu_seconds": live_cpu,
        "live_step_cpu_seconds": live_cpu,
        "first_step_cpu_seconds": max(0.0, step_cpu[0] - first_branch_cpu) if step_cpu else 0.0,
        "steady_live_step_cpu_seconds": max(0.0, steady),
        "steady_live_decisions": max(0, len(step_cpu) - 1),
        "construction": construction,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": time.perf_counter() - start_wall,
    }


def _verify_metrics() -> dict[str, str]:
    metric = Path(__file__).with_name("METRICS.md")
    digest = _sha(metric)
    expected = os.environ.get("BL_DENSITY_METRICS_SHA256")
    commit = os.environ.get("BL_DENSITY_COMMIT", "HEAD")
    if expected and expected != digest:
        raise RuntimeError("METRICS.md SHA does not match supervisor binding")
    try:
        frozen = subprocess.check_output(
            ["git", "show", f"{commit}:research/bl_density/METRICS.md"], text=False
        )
    except subprocess.CalledProcessError as exc:
        if expected:
            frozen = metric.read_bytes()
        else:
            raise RuntimeError("committed METRICS.md unavailable before gameplay") from exc
    if hashlib.sha256(frozen).hexdigest() != digest:
        raise RuntimeError("working metric contract differs from committed binding")
    amendment = Path(__file__).with_name("METRICS_AMENDMENT_01.md")
    amendment_digest = _sha(amendment)
    expected_amendment = os.environ.get("BL_DENSITY_AMENDMENT_SHA256")
    if expected_amendment and expected_amendment != amendment_digest:
        raise RuntimeError("METRICS_AMENDMENT_01.md SHA does not match supervisor binding")
    try:
        frozen_amendment = subprocess.check_output(
            ["git", "show", f"{commit}:research/bl_density/METRICS_AMENDMENT_01.md"], text=False
        )
    except subprocess.CalledProcessError:
        if expected_amendment:
            frozen_amendment = amendment.read_bytes()
        else:
            raise RuntimeError("committed metric amendment unavailable before gameplay")
    if hashlib.sha256(frozen_amendment).hexdigest() != amendment_digest:
        raise RuntimeError("working metric amendment differs from committed binding")
    capacity = Path(__file__).with_name("METRICS_AMENDMENT_02.md")
    capacity_digest = _sha(capacity)
    expected_capacity = os.environ.get("BL_DENSITY_CAPACITY_AMENDMENT_SHA256")
    if expected_capacity and expected_capacity != capacity_digest:
        raise RuntimeError("METRICS_AMENDMENT_02.md SHA does not match supervisor binding")
    try:
        frozen_capacity = subprocess.check_output(
            ["git", "show", f"{commit}:research/bl_density/METRICS_AMENDMENT_02.md"], text=False
        )
    except subprocess.CalledProcessError:
        if expected_capacity:
            frozen_capacity = capacity.read_bytes()
        else:
            raise RuntimeError("committed capacity amendment unavailable before gameplay")
    if hashlib.sha256(frozen_capacity).hexdigest() != capacity_digest:
        raise RuntimeError("working capacity amendment differs from committed binding")
    amendment03 = Path(__file__).with_name("METRICS_AMENDMENT_03.md")
    amendment03_digest = _sha(amendment03)
    expected_amendment03 = os.environ.get("BL_DENSITY_ROUND14_AMENDMENT_SHA256")
    if expected_amendment03 and expected_amendment03 != amendment03_digest:
        raise RuntimeError("METRICS_AMENDMENT_03.md SHA does not match supervisor binding")
    try:
        frozen_amendment03 = subprocess.check_output(
            ["git", "show", f"{commit}:research/bl_density/METRICS_AMENDMENT_03.md"], text=False
        )
    except subprocess.CalledProcessError:
        if expected_amendment03:
            frozen_amendment03 = amendment03.read_bytes()
        else:
            raise RuntimeError("committed amendment 03 unavailable before gameplay")
    if hashlib.sha256(frozen_amendment03).hexdigest() != amendment03_digest:
        raise RuntimeError("working amendment 03 differs from committed binding")
    return {
        "metrics_sha256": digest,
        "metrics_amendment_sha256": amendment_digest,
        "metrics_capacity_amendment_sha256": capacity_digest,
        "metrics_round14_amendment_sha256": amendment03_digest,
        "metrics_commit": commit,
    }


def planned_jobs(
    context: dict[str, Any],
    *,
    b8: Path,
    arms: Sequence[str],
    mixes: Sequence[str],
    episodes: int,
    horizon: int,
    tiny: bool,
    deadline: float | None,
) -> list[dict[str, Any]]:
    """Materialize each entire cell before execution, preserving pool rotation."""
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

    jobs = []
    for arm in arms:
        for mix in mixes:
            count = episodes if tiny else 12 if arm == "A" else 6
            seeds = [_seed(arm, mix, index) for index in range(count)]
            plans = planned_rosters(context, seeds, [mix] * len(seeds))
            for index, plan in enumerate(plans):
                jobs.append(
                    {
                        "b8": str(b8),
                        "arm": arm,
                        "mix": mix,
                        "index": index,
                        "plan": plan,
                        "horizon": horizon,
                        "tiny": tiny,
                        "deadline": deadline,
                    }
                )
    if len({job["plan"]["seed"] for job in jobs}) != len(jobs):
        raise RuntimeError("density namespace has an internal seed collision")
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b8", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--episodes-per-cell", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--mixes", default=",".join(MIXES))
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args(argv)
    if args.tiny:
        args.horizon, args.episodes_per_cell, args.workers = (
            min(args.horizon, 8),
            min(args.episodes_per_cell, 1),
            1,
        )
    elif args.horizon != HORIZON or args.episodes_per_cell != 6:
        raise SystemExit("production allocation is frozen at H5000, A=12/mix, others=6/mix")
    elif args.workers != 6:
        raise SystemExit(
            "production requires six CPU workers; local default remains bounded at two"
        )
    if args.workers < 1 or args.workers > 6 or args.episodes_per_cell < 1:
        raise SystemExit("invalid bounded worker/allocation request")
    if (
        args.horizon < 1
        or args.max_seconds is not None
        and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0)
    ):
        raise SystemExit("horizon and work bound must be finite and positive")
    binding = _verify_metrics()
    arms, mixes = tuple(args.arms.split(",")), tuple(args.mixes.split(","))
    if _sha(args.b8) != "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8":
        raise SystemExit("--b8 does not match the frozen B8 s3 checkpoint digest")
    if set(arms) - set(ARMS) or set(mixes) - set(MIXES):
        raise SystemExit("unknown arm or mix")
    if len(set(arms)) != len(arms) or len(set(mixes)) != len(mixes):
        raise SystemExit("duplicate arm or mix")
    if not args.tiny and (arms != ARMS or mixes != MIXES):
        raise SystemExit("production requires all frozen arms and mixes")
    if args.out.exists():
        raise SystemExit("--out must be a fresh directory")
    args.out.mkdir(parents=True)
    started = time.monotonic()
    parent_cpu = time.process_time()
    records = []
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    jobs = planned_jobs(
        gid._context(1),
        b8=args.b8,
        arms=arms,
        mixes=mixes,
        episodes=args.episodes_per_cell,
        horizon=args.horizon,
        tiny=bool(args.tiny),
        deadline=(started + args.max_seconds) if args.max_seconds is not None else None,
    )
    child_before = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn")
    ) as pool:
        pending: dict[Any, dict[str, Any]] = {}
        cursor = 0
        while cursor < len(jobs) or pending:
            while cursor < len(jobs) and len(pending) < args.workers:
                if args.max_seconds is not None and time.monotonic() - started >= args.max_seconds:
                    cursor = len(jobs)
                    break
                future = pool.submit(run_episode, jobs[cursor])
                pending[future] = jobs[cursor]
                cursor += 1
            if not pending:
                break
            done, _ = concurrent.futures.wait(
                pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED
            )
            if not done:
                print(
                    json.dumps(
                        {
                            "event": "bl_density_progress",
                            "completed": len(records),
                            "scheduled": len(jobs),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
                continue
            for future in done:
                pending.pop(future)
                record = future.result()
                record.update(binding)
                records.append(record)
                export_json(
                    args.out / "episodes" / f"{record['episode_id'].replace('/', '_')}.json", record
                )
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "event": "bl_density_progress",
                            "completed": len(records),
                            "scheduled": len(jobs),
                            "elapsed_seconds": elapsed,
                            "eta_seconds": elapsed / len(records) * (len(jobs) - len(records)),
                        }
                    ),
                    flush=True,
                )
    # Completion order is deliberately unconstrained by process scheduling;
    # every downstream descriptive calculation receives the fixed job order.
    records.sort(key=lambda record: str(record["episode_id"]))
    status = (
        "COMPLETE"
        if len(records) == len(jobs) and all(r["complete"] for r in records)
        else "INCOMPLETE"
    )
    child_after = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    )
    episode_cpu = sum(float(r["cpu_seconds"]) for r in records)
    from research.bl_density.analysis import finalize_cpu, summarize

    # Fit/KS once, then charge its measured CPU before the lightweight repricing.
    analysis_started = time.process_time()
    analysis = summarize(records, FEATURE_NAMES, run_status=status)
    analysis_cpu = time.process_time() - analysis_started
    parent_total_cpu = time.process_time()
    end_to_end_cpu = parent_total_cpu + child_after
    overhead = max(0.0, end_to_end_cpu - episode_cpu)
    analysis = finalize_cpu(analysis, records, overhead)
    summary = {
        "schema": "bl-density-run/v2",
        "status": status,
        "development_only": True,
        "tiny_preflight": bool(args.tiny),
        **binding,
        "b8_sha256": _sha(args.b8),
        "scheduled_episodes": len(jobs),
        "completed_episodes": len(records),
        "records": [r["episode_id"] for r in records],
        "parent_analysis_cpu_seconds": analysis_cpu,
        "parent_cpu_seconds": parent_total_cpu,
        "parent_precollection_cpu_seconds": parent_cpu,
        "end_to_end_cpu_seconds": end_to_end_cpu,
        "doom_moments_per_end_to_end_cpu_second": (
            analysis["aggregate_complete_episode_metrics"]["doom_moments"] / end_to_end_cpu
            if end_to_end_cpu
            else None
        ),
        "overhead_attribution": (
            "non-episode CPU equally per attempted episode; partial CPU charged to its own cell"
        ),
        "child_cpu_seconds": child_after - child_before,
        "worker_startup_import_cpu_seconds": max(0.0, child_after - child_before - episode_cpu),
        **analysis,
    }
    export_json(args.out / "summary.json", summary)
    return 0 if status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
