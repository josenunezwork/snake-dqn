"""Stage 0b bare B8 death-cause audit on the 48 Watch6-v2 H=6,000 probe worlds."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from research.b8_death_census_20261014.census import ordered_cells
from research.bl_k1.native import RowStats, branch_scope, configure
from research.bl_served_search import ceiling
from research.bl_served_search.ruler_operator import _clone_root
from research.hero_stage0.protocol import (
    CAUSE_CLASSES,
    COUNT,
    HORIZON,
    LIVE_CAUSE_TO_CAUSE_CLASS,
    LOOKBACK_FRAMES,
    NAMESPACE,
    SIM_CODE_TO_CAUSE_CLASS,
    allocation,
)
from research.t3_rollout_labels_20261015.bank_engine import (
    choose_and_step,
    prepared_q,
    step_prepared,
)
from research.t15_mixed_roster_20261017.episodes import _assert_mixed
from research.watch_replay.product import open_watch
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.eval_engine import _dispatch_actions
from src.simd_env.v8_nb import v8_decide

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-hero-stage0")
K_STEPS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10, 12, 16)
V8_LAMBDA = 8.0


def _enforce_single_thread_env() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[var] = "1"
    os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
    os.environ.setdefault("SNAKE_CKPT_DIR", str(REPO_ROOT / "saved_snakes"))
    torch.set_num_threads(1)


def _opponent_proximity(sim: Any) -> dict[str, Any]:
    """Compute Manhattan and Chebyshev distances from hero head to alive opponent heads/bodies."""
    hx, hy = (int(v) for v in sim.bodies[0, 0, int(sim.head_ptr[0, 0])])
    min_head_l1: int | None = None
    min_body_l1: int | None = None
    min_head_linf: int | None = None
    min_body_linf: int | None = None
    opponents: list[dict[str, Any]] = []

    for slot in range(1, int(sim.S)):
        if not bool(sim.alive[0, slot]):
            continue
        ox, oy = (int(v) for v in sim.bodies[0, slot, int(sim.head_ptr[0, slot])])
        h_l1 = abs(ox - hx) + abs(oy - hy)
        h_linf = max(abs(ox - hx), abs(oy - hy))
        cells = ordered_cells(sim, 0, slot)
        dx = np.abs(cells[:, 0] - hx)
        dy = np.abs(cells[:, 1] - hy)
        b_l1 = int(np.min(dx + dy)) if len(cells) else h_l1
        b_linf = int(np.min(np.maximum(dx, dy))) if len(cells) else h_linf

        if min_head_l1 is None or h_l1 < min_head_l1:
            min_head_l1 = h_l1
        if min_body_l1 is None or b_l1 < min_body_l1:
            min_body_l1 = b_l1
        if min_head_linf is None or h_linf < min_head_linf:
            min_head_linf = h_linf
        if min_body_linf is None or b_linf < min_body_linf:
            min_body_linf = b_linf

        opponents.append(
            {
                "slot": slot,
                "length": int(sim.length[0, slot]),
                "head": [ox, oy],
                "head_l1": h_l1,
                "body_l1": b_l1,
                "head_linf": h_linf,
                "body_linf": b_linf,
            }
        )

    return {
        "hero_head": [hx, hy],
        "alive_opponents": len(opponents),
        "min_opp_head_l1": min_head_l1,
        "min_opp_body_l1": min_body_l1,
        "min_opp_head_linf": min_head_linf,
        "min_opp_body_linf": min_body_linf,
        "opp_head_within_k": {
            str(k): bool(min_head_l1 is not None and min_head_l1 <= k) for k in K_STEPS
        },
        "opp_body_within_k": {
            str(k): bool(min_body_l1 is not None and min_body_l1 <= k) for k in K_STEPS
        },
        "opponents": opponents,
    }


def _pocket_boundary_composition(sim: Any, max_cells: int = 200) -> dict[str, Any]:
    """Flood-fill empty cells around hero head and attribute bounding obstacles."""
    owner = sim._owner[0]
    gh, gw = int(sim.grid_h), int(sim.grid_w)
    hx, hy = (int(v) for v in sim.bodies[0, 0, int(sim.head_ptr[0, 0])])
    opp_heads: set[tuple[int, int]] = set()
    for slot in range(1, int(sim.S)):
        if bool(sim.alive[0, slot]):
            ox, oy = (int(v) for v in sim.bodies[0, slot, int(sim.head_ptr[0, slot])])
            opp_heads.add((ox, oy))

    visited: set[tuple[int, int]] = {(hx, hy)}
    queue: deque[tuple[int, int]] = deque([(hx, hy)])
    empty_count = 0
    wall_bounds: set[tuple[int, int]] = set()
    self_bounds: set[tuple[int, int]] = set()
    opp_head_bounds: set[tuple[int, int]] = set()
    opp_body_bounds: set[tuple[int, int]] = set()

    while queue and empty_count < max_cells:
        cx, cy = queue.popleft()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if not (0 <= nx < gw and 0 <= ny < gh):
                wall_bounds.add((nx, ny))
                continue
            occ = int(owner[ny, nx])
            if occ < 0:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    empty_count += 1
                    if empty_count < max_cells:
                        queue.append((nx, ny))
            elif occ == 0:
                if (nx, ny) != (hx, hy):
                    self_bounds.add((nx, ny))
            else:
                if (nx, ny) in opp_heads:
                    opp_head_bounds.add((nx, ny))
                else:
                    opp_body_bounds.add((nx, ny))

    n_self = len(self_bounds)
    n_opp = len(opp_head_bounds) + len(opp_body_bounds)
    n_wall = len(wall_bounds)
    snake_bounds = n_self + n_opp
    self_frac = float(n_self / snake_bounds) if snake_bounds > 0 else 0.0
    opp_frac = float(n_opp / snake_bounds) if snake_bounds > 0 else 0.0
    if snake_bounds == 0:
        dominant = "wall"
    elif n_self >= n_opp:
        dominant = "self_body"
    else:
        dominant = "opponent"

    return {
        "reachable_empty_capped": empty_count,
        "cap": max_cells,
        "wall_boundary_cells": n_wall,
        "self_body_boundary_cells": n_self,
        "opp_head_boundary_cells": len(opp_head_bounds),
        "opp_body_boundary_cells": len(opp_body_bounds),
        "snake_boundary_self_frac": round(self_frac, 4),
        "snake_boundary_opp_frac": round(opp_frac, 4),
        "dominant_snake_blocker": dominant,
    }


def _ensure_snap_v8(snap: dict[str, Any]) -> dict[str, Any]:
    """Lazily compute v8 shadow diagnostics on a buffered decision snapshot."""
    if "v8_action" not in snap:
        sim = snap["root_clone"].sim
        q = np.asarray(snap["q"], dtype=np.float32)
        advisory = np.asarray(snap["advisory"], dtype=bool)
        bare = int(snap["bare"])
        v8_res = v8_decide(
            sim,
            np.array([[0, 0]], dtype=np.int64),
            q.reshape(1, 6),
            advisory.reshape(1, 6),
            V8_LAMBDA,
            features=True,
        )
        area_vec = v8_res.area[0].copy()
        best_safe_area = int(np.where(advisory, area_vec, 0).max()) if advisory.any() else 0
        need = int(v8_res.need[0])
        snap["v8_action"] = int(v8_res.action[0])
        snap["v8_outcome"] = int(v8_res.outcome[0])
        snap["need"] = need
        snap["best_safe_area"] = best_safe_area
        snap["taken_area"] = int(area_vec[bare]) if advisory[bare] else 0
        snap["enclosed"] = bool(best_safe_area < need)
    return snap


def _ensure_snap_geometry(snap: dict[str, Any]) -> dict[str, Any]:
    """Lazily compute opponent proximity and pocket boundary on a buffered decision snapshot."""
    _ensure_snap_v8(snap)
    if "proximity" not in snap:
        sim = snap["root_clone"].sim
        snap["proximity"] = _opponent_proximity(sim)
        snap["pocket_boundary"] = _pocket_boundary_composition(sim)
    return snap


def _evaluate_frame_counterfactuals(
    snap: dict[str, Any],
    h_to_death: int,
    *,
    evaluate_bare: bool = False,
) -> dict[int, dict[str, Any]]:
    """Evaluate all 6 actions from ``snap['root_clone']`` for ``h_to_death`` steps."""
    _ensure_snap_v8(snap)
    outcomes: dict[int, dict[str, Any]] = {}
    bare = int(snap["bare"])
    resolved = snap["resolved"]
    advisory = snap["advisory"]

    for action in range(6):
        is_resolved_legal = bool(resolved[action])
        is_advisory_legal = bool(advisory[action])
        if action == bare and not evaluate_bare:
            outcomes[action] = {
                "action": action,
                "is_bare": True,
                "resolved_legal": is_resolved_legal,
                "advisory_legal": is_advisory_legal,
                "survives_to_death": False,
                "survives_open_at_death": False,
                "frames_executed": h_to_death,
                "end_best_safe_area": 0,
                "end_need": int(snap["need"]),
                "end_length": 0,
                "death_cause_code": None,
            }
            continue
        if not is_resolved_legal and action != bare:
            outcomes[action] = {
                "action": action,
                "is_bare": False,
                "resolved_legal": False,
                "advisory_legal": False,
                "survives_to_death": False,
                "survives_open_at_death": False,
                "frames_executed": 0,
                "end_best_safe_area": 0,
                "end_need": int(snap["need"]),
                "end_length": 0,
                "death_cause_code": None,
            }
            continue

        branch = _clone_root(snap["root_clone"])
        branch.sim._policy_phase_prepared = True
        step_prepared(branch, int(action))
        executed = 1
        while executed < h_to_death and bool(branch.sim.alive[0, 0]):
            choose_and_step(branch, capture_obs=False)
            executed += 1

        alive_end = bool(branch.sim.alive[0, 0])
        survives = alive_end and executed == h_to_death
        death_code = None if alive_end else int(branch.sim.get_death_cause()[0, 0])
        end_len = int(branch.sim.length[0, 0]) if alive_end else 0
        end_best_area = 0
        end_need = int(snap["need"])
        survives_open = False

        if survives:
            adv_end = branch.sim.get_advisory_action_mask()[0, 0].astype(bool)
            if adv_end.any():
                res_end = v8_decide(
                    branch.sim,
                    np.array([[0, 0]], dtype=np.int64),
                    np.zeros((1, 6), dtype=np.float32),
                    adv_end.reshape(1, 6),
                    V8_LAMBDA,
                    features=True,
                )
                end_best_area = int(np.where(adv_end, res_end.area[0], 0).max())
                end_need = int(res_end.need[0])
                survives_open = bool(end_best_area >= end_need)
            else:
                end_need = max(32, min(160, 2 * end_len))
                survives_open = False

        outcomes[action] = {
            "action": action,
            "is_bare": action == bare,
            "resolved_legal": is_resolved_legal,
            "advisory_legal": is_advisory_legal,
            "survives_to_death": bool(survives),
            "survives_open_at_death": bool(survives_open),
            "frames_executed": int(executed),
            "end_best_safe_area": int(end_best_area),
            "end_need": int(end_need),
            "end_length": int(end_len),
            "death_cause_code": death_code,
        }
    return outcomes


def _analyze_hero_death(
    *,
    world_index: int,
    seed: int,
    death_index_in_world: int,
    life: int,
    t_death: int,
    sim_cause_code: int,
    live_cause_str: str,
    life_snaps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Perform the 300-frame lookback and classify a single bare B8 hero death."""
    t0 = time.perf_counter()
    fatal_snap = life_snaps[-1]
    if int(fatal_snap["frame"]) != t_death:
        raise RuntimeError(f"fatal snapshot frame {fatal_snap['frame']} != t_death {t_death}")

    cause_from_sim = SIM_CODE_TO_CAUSE_CLASS.get(sim_cause_code)
    cause_from_live = LIVE_CAUSE_TO_CAUSE_CLASS.get(live_cause_str)
    if cause_from_sim is None or cause_from_sim != cause_from_live:
        raise RuntimeError(
            f"Watch/Grid death cause mismatch at frame {t_death}: "
            f"sim={sim_cause_code} ({cause_from_sim}) vs "
            f"live={live_cause_str!r} ({cause_from_live})"
        )
    cause_class = cause_from_sim

    # Compute consecutive enclosed frames (v8 best_safe_area < need) ending at t_death
    v8_trap_run = 0
    for s in reversed(life_snaps):
        _ensure_snap_v8(s)
        if bool(s["enclosed"]):
            v8_trap_run += 1
        else:
            break

    # Shadow v8 agreement over last 1..10 frames
    for s in life_snaps[-10:]:
        _ensure_snap_v8(s)
    v8_differs_flags = [int(s["v8_action"]) != int(s["bare"]) for s in life_snaps[-10:]]
    v8_differs_within_k = {str(k): bool(any(v8_differs_flags[-k:])) for k in range(1, 11)}

    # Exact-sim check of the 6 actions over buffered lookback (up to 300 frames, H to death)
    max_lookback_idx = min(LOOKBACK_FRAMES - 1, len(life_snaps) - 1)
    d_last_surviving: int | None = None
    d_last_open_surviving: int | None = None
    surviving_actions_at_last: list[int] = []
    open_surviving_actions_at_last: list[int] = []
    lookback_records: dict[str, Any] = {}
    branch_steps_total = 0
    stats = RowStats()

    with configure("kernel", stats), branch_scope("mixed"):
        for d in range(max_lookback_idx + 1):
            snap = life_snaps[-1 - d]
            h_to_death = d + 1  # steps from frame (t_death - d) through t_death inclusive
            outcomes = _evaluate_frame_counterfactuals(snap, h_to_death, evaluate_bare=(d == 0))
            branch_steps_total += sum(
                int(o["frames_executed"])
                for a, o in outcomes.items()
                if d == 0 or a != int(snap["bare"])
            )
            if d == 0:
                bare_out = outcomes[int(snap["bare"])]
                if bare_out["survives_to_death"] or bare_out["death_cause_code"] != sim_cause_code:
                    raise RuntimeError(
                        f"d=0 bare control mismatch at frame {t_death}: "
                        f"expected death code {sim_cause_code}, got {bare_out}"
                    )

            alt_survivors = [
                a
                for a in range(6)
                if a != int(snap["bare"])
                and outcomes[a]["resolved_legal"]
                and outcomes[a]["survives_to_death"]
            ]
            alt_open_survivors = [
                a
                for a in range(6)
                if a != int(snap["bare"])
                and outcomes[a]["resolved_legal"]
                and outcomes[a]["survives_open_at_death"]
            ]

            if (
                d <= 10
                or (d_last_surviving is None and alt_survivors)
                or (d_last_open_surviving is None and alt_open_survivors)
            ):
                _ensure_snap_geometry(snap)
                lookback_records[str(d)] = {
                    "d_frames_before_fatal": d,
                    "h_steps_to_death": h_to_death,
                    "frame": int(snap["frame"]),
                    "length": int(snap["length"]),
                    "bare_action": int(snap["bare"]),
                    "v8_action": int(snap["v8_action"]),
                    "advisory_legal": [int(a) for a in np.flatnonzero(snap["advisory"])],
                    "resolved_legal": [int(a) for a in np.flatnonzero(snap["resolved"])],
                    "alt_surviving_actions": alt_survivors,
                    "alt_open_surviving_actions": alt_open_survivors,
                    "v8_action_survives_to_death": bool(
                        outcomes[int(snap["v8_action"])]["survives_to_death"]
                    ),
                    "v8_action_survives_open": bool(
                        outcomes[int(snap["v8_action"])]["survives_open_at_death"]
                    ),
                    "min_opp_head_l1": snap["proximity"]["min_opp_head_l1"],
                    "min_opp_body_l1": snap["proximity"]["min_opp_body_l1"],
                    "pocket_boundary": snap["pocket_boundary"],
                    "outcomes": {str(k): v for k, v in outcomes.items()},
                }

            if d_last_surviving is None and alt_survivors:
                d_last_surviving = d
                surviving_actions_at_last = alt_survivors
            if d_last_open_surviving is None and alt_open_survivors:
                d_last_open_surviving = d
                open_surviving_actions_at_last = alt_open_survivors

            if d >= 10 and d_last_surviving is not None and d_last_open_surviving is not None:
                break

    _ensure_snap_geometry(fatal_snap)
    last_surv_snap = (
        _ensure_snap_geometry(life_snaps[-1 - d_last_surviving])
        if d_last_surviving is not None
        else None
    )
    last_open_snap = (
        _ensure_snap_geometry(life_snaps[-1 - d_last_open_surviving])
        if d_last_open_surviving is not None
        else None
    )

    # Steps to death: 1 means fatal frame (d=0), 2..8 means 1..7 frames before fatal (d=1..7)
    steps_from_last_surviving = d_last_surviving + 1 if d_last_surviving is not None else None
    steps_from_last_open_surviving = (
        d_last_open_surviving + 1 if d_last_open_surviving is not None else None
    )

    # Topological cause refinement: if enclosed for >=3 frames, attribute to dominant pocket blocker
    fatal_boundary = fatal_snap["pocket_boundary"]
    if v8_trap_run >= 3:
        if fatal_boundary["self_body_boundary_cells"] >= (
            fatal_boundary["opp_head_boundary_cells"] + fatal_boundary["opp_body_boundary_cells"]
        ):
            topological_cause = "self-body enclosure"
        else:
            topological_cause = "opponent body"
    else:
        topological_cause = cause_class

    length = int(fatal_snap["length"])
    return {
        "world_index": world_index,
        "seed": seed,
        "death_index_in_world": death_index_in_world,
        "life": life,
        "t_death": t_death,
        "life_frames_alive": len(life_snaps),
        "length": length,
        "length_regime": "L<600" if length < 600 else "L>=600",
        "sim_cause_code": sim_cause_code,
        "live_cause": live_cause_str,
        "cause_class": cause_class,
        "topological_cause_class": topological_cause,
        "fatal_bare_action": int(fatal_snap["bare"]),
        "fatal_bare_is_boost": int(fatal_snap["bare"]) >= 3,
        "fatal_v8_action": int(fatal_snap["v8_action"]),
        "fatal_v8_outcome": int(fatal_snap["v8_outcome"]),
        "fatal_advisory_any": bool(np.any(fatal_snap["advisory"])),
        "fatal_advisory_count": int(np.sum(fatal_snap["advisory"])),
        "fatal_best_safe_area": int(fatal_snap["best_safe_area"]),
        "fatal_taken_area": int(fatal_snap["taken_area"]),
        "fatal_need": int(fatal_snap["need"]),
        "fatal_enclosed": bool(fatal_snap["enclosed"]),
        "v8_trap_run": int(v8_trap_run),
        "v8_differs_within_k": v8_differs_within_k,
        "lookback_window_available": max_lookback_idx + 1,
        "frames_before_fatal_last_surviving": d_last_surviving,
        "steps_to_death_from_last_surviving": steps_from_last_surviving,
        "surviving_actions_at_last_surviving": surviving_actions_at_last,
        "v8_survives_at_last_surviving": (
            bool(int(last_surv_snap["v8_action"]) in surviving_actions_at_last)
            if last_surv_snap is not None
            else False
        ),
        "frames_before_fatal_last_open_surviving": d_last_open_surviving,
        "steps_to_death_from_last_open_surviving": steps_from_last_open_surviving,
        "open_surviving_actions_at_last_open": open_surviving_actions_at_last,
        "v8_survives_at_last_open_surviving": (
            bool(int(last_open_snap["v8_action"]) in open_surviving_actions_at_last)
            if last_open_snap is not None
            else False
        ),
        "proximity_at_death": fatal_snap["proximity"],
        "proximity_at_last_surviving": (
            last_surv_snap["proximity"] if last_surv_snap is not None else None
        ),
        "proximity_at_last_open_surviving": (
            last_open_snap["proximity"] if last_open_snap is not None else None
        ),
        "pocket_boundary_at_death": fatal_boundary,
        "pocket_boundary_at_last_surviving": (
            last_surv_snap["pocket_boundary"] if last_surv_snap is not None else None
        ),
        "pocket_boundary_at_last_open_surviving": (
            last_open_snap["pocket_boundary"] if last_open_snap is not None else None
        ),
        "lookback_records": lookback_records,
        "branch_steps_evaluated": int(branch_steps_total),
        "analysis_wall_seconds": round(time.perf_counter() - t0, 4),
    }


def run_world_death_audit(
    world_row: dict[str, Any],
    horizon: int = HORIZON,
    worlds_dir: Path | None = None,
) -> dict[str, Any]:
    """Run bare b8-s3 on one Watch6-v2 world with live/Grid mirror and audit every hero death."""
    _enforce_single_thread_env()
    world_index = int(world_row["world_index"])
    seed = int(world_row["seed"])
    w_path = worlds_dir / f"probe-world-{world_index:02d}.json" if worlds_dir is not None else None
    if w_path is not None and w_path.is_file():
        cached = json.loads(w_path.read_text(encoding="utf-8"))
        if int(cached.get("seed", -1)) == seed and int(cached.get("horizon", -1)) == horizon:
            return cached

    t_start = time.perf_counter()

    session = open_watch({"seed": seed}, 24, "bare")
    world = session.world
    if not world.sim.allow_respawn or any(
        not getattr(s, "auto_respawn", True) for s in session.live.game.snakes
    ):
        session.close()
        raise ValueError("Watch6-v2 requires continuous respawn in live and Grid mirror")
    if getattr(session.live, "pathb", None) is not None:
        session.close()
        raise ValueError("Stage 0b bare audit refuses Path B assistance")
    _assert_mixed(session.live, 6)

    life = 1
    was_alive = True
    life_ring: deque[dict[str, Any]] = deque(maxlen=LOOKBACK_FRAMES)
    deaths: list[dict[str, Any]] = []
    l_lt_600_alive_frames = 0
    l_ge_600_alive_frames = 0
    dead_frames = 0
    sum_mass = 0

    import contextlib

    try:
        with contextlib.nullcontext():
            for frame in range(1, horizon + 1):
                selected: dict[str, Any] = {}

                def choose(sim: Any) -> np.ndarray:
                    nonlocal life
                    world.runtime.prepare(sim)
                    active = bool(sim.alive[0, 0])
                    if active and not was_alive:
                        life += 1
                        life_ring.clear()
                    actions = np.ones((1, sim.S), dtype=np.int64)
                    if active:
                        root_clone = _clone_root(world)
                        root_clone.sim._policy_phase_prepared = True
                        obs = build_ego_raster(
                            sim,
                            EgoRasterConfig(version="bv8"),
                            backend="numba",
                            rows=np.array([[0, 0]], dtype=np.int64),
                        )
                        q = np.asarray(prepared_q(world, obs), dtype=np.float32)
                        advisory = sim.get_advisory_action_mask()[0, 0].astype(bool).copy()
                        resolved = sim.get_resolved_action_mask()[0, 0].astype(bool).copy()
                        _, bare = ceiling._legal(q, advisory)
                        if not np.isfinite(q).all() or not 0 <= bare < 6:
                            raise ValueError("nonfinite Q or illegal bare action")

                        length = int(sim.length[0, 0])
                        snap = {
                            "frame": frame,
                            "life": life,
                            "length": length,
                            "bare": bare,
                            "q": q.astype(float).tolist(),
                            "advisory": advisory,
                            "resolved": resolved,
                            "root_clone": root_clone,
                        }
                        life_ring.append(snap)
                        selected.update(
                            frame=frame,
                            life=life,
                            length=length,
                            bare=bare,
                            chosen=bare,
                            q=snap["q"],
                        )

                    _dispatch_actions(
                        sim,
                        sim.get_resolved_action_mask(),
                        actions,
                        [world.hero],
                        [world.opponents],
                    )
                    if active:
                        if int(actions[0, 0]) != selected["bare"]:
                            raise ValueError("bare dispatch/Q mismatch")
                        actions[0, 0] = selected["chosen"]
                    return actions

                actions_holder: dict[str, Any] = {}

                def choose_record(sim: Any) -> np.ndarray:
                    acts = choose(sim)
                    actions_holder["actions"] = acts.copy()
                    return acts

                world.sim.step_with_policy(choose_record)
                world.runtime.observe_step(world.sim)

                # Step and audit live Watch session mirror
                actual_decisions: list[int] = []
                hero = session.live.game.snakes[0]
                original_update = hero.update
                original_q = session.live.policy._next_dispatch_q

                def observed_update(*args: Any, **kwargs: Any) -> Any:
                    actual_decisions.append(int(hero.length))
                    return original_update(*args, **kwargs)

                def observed_q(*args: Any, **kwargs: Any) -> Any:
                    value = original_q(*args, **kwargs)
                    if not selected or not np.array_equal(
                        value.detach().cpu().numpy().reshape(6),
                        np.asarray(selected["q"], dtype=np.float32),
                    ):
                        raise ValueError("live/mirror B8 Q mismatch")
                    return value

                hero.update = observed_update
                session.live.policy._next_dispatch_q = observed_q
                try:
                    session._step_live(actions_holder["actions"])
                finally:
                    hero.update = original_update
                    session.live.policy._next_dispatch_q = original_q

                if actual_decisions != ([selected["length"]] if selected else []):
                    raise ValueError("live/mirror actual hero decision mismatch")

                hero = session.live.game.snakes[0]
                live_death = int(hero.id) in session.live.game.frame_death_causes
                grid_death = bool(
                    world.sim.get_transition_valid()[0, 0] and not world.sim.alive[0, 0]
                )
                if live_death != grid_death:
                    raise ValueError("Watch/mirror death event mismatch")
                if selected:
                    if int(hero.last_action) != selected["chosen"]:
                        raise ValueError("live delivered action mismatch")
                    if int(selected["length"]) < 600:
                        l_lt_600_alive_frames += 1
                    else:
                        l_ge_600_alive_frames += 1

                was_alive = bool(hero.is_alive)
                if was_alive:
                    sum_mass += int(hero.length)
                else:
                    dead_frames += 1

                if live_death:
                    sim_cause_code = int(world.sim.get_death_cause()[0, 0])
                    live_cause_str = str(session.live.game.frame_death_causes[int(hero.id)])
                    death_rec = _analyze_hero_death(
                        world_index=world_index,
                        seed=seed,
                        death_index_in_world=len(deaths),
                        life=life,
                        t_death=frame,
                        sim_cause_code=sim_cause_code,
                        live_cause_str=live_cause_str,
                        life_snaps=list(life_ring),
                    )
                    deaths.append(death_rec)
                    life_ring.clear()
    finally:
        session.close()

    elapsed = time.perf_counter() - t_start
    l_lt_600_deaths = sum(1 for d in deaths if d["length"] < 600)
    l_ge_600_deaths = sum(1 for d in deaths if d["length"] >= 600)
    res = {
        "world_index": world_index,
        "seed": seed,
        "horizon": horizon,
        "mirror_identity_frames": horizon,
        "mean_mass": round(float(sum_mass) / float(horizon), 4),
        "l_lt_600_alive_frames": l_lt_600_alive_frames,
        "l_ge_600_alive_frames": l_ge_600_alive_frames,
        "dead_frames": dead_frames,
        "total_deaths": len(deaths),
        "l_lt_600_deaths": l_lt_600_deaths,
        "l_ge_600_deaths": l_ge_600_deaths,
        "wall_seconds": round(elapsed, 3),
        "deaths": deaths,
    }
    if w_path is not None:
        tmp_path = w_path.with_suffix(".tmp.json")
        tmp_path.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(w_path)
    print(
        f"[Stage0b] world={world_index:02d} seed={seed} "
        f"deaths={res['total_deaths']} (L<600: {l_lt_600_deaths}, "
        f"L>=600: {l_ge_600_deaths}) mass={res['mean_mass']:.1f} "
        f"wall={res['wall_seconds']:.1f}s",
        flush=True,
    )
    return res


def _step_bucket(steps: int | None) -> str:
    if steps is None:
        return ">300_or_unrescuable"
    if steps == 1:
        return "1_step_(d=0)"
    if 2 <= steps <= 8:
        return "2_to_8_steps_(d=1..7)"
    if 9 <= steps <= 20:
        return "9_to_20_steps_(d=8..19)"
    if 21 <= steps <= 50:
        return "21_to_50_steps_(d=20..49)"
    if 51 <= steps <= 100:
        return "51_to_100_steps_(d=50..99)"
    return "101_to_300_steps_(d=100..299)"


STEP_BUCKET_ORDER: tuple[str, ...] = (
    "1_step_(d=0)",
    "2_to_8_steps_(d=1..7)",
    "9_to_20_steps_(d=8..19)",
    "21_to_50_steps_(d=20..49)",
    "51_to_100_steps_(d=50..99)",
    "101_to_300_steps_(d=100..299)",
    ">300_or_unrescuable",
)


def _quantiles(values: Sequence[int | float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "p10": round(float(np.percentile(arr, 10)), 3),
        "p25": round(float(np.percentile(arr, 25)), 3),
        "median": round(float(np.percentile(arr, 50)), 3),
        "p75": round(float(np.percentile(arr, 75)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "max": round(float(np.max(arr)), 3),
    }


def _summarize_regime(
    deaths: Sequence[dict[str, Any]],
    world_counts: Sequence[int],
    alive_frames: int,
) -> dict[str, Any]:
    n = len(deaths)
    cause_counts = Counter(d["cause_class"] for d in deaths)
    topo_cause_counts = Counter(d["topological_cause_class"] for d in deaths)

    surv_buckets = Counter(_step_bucket(d["steps_to_death_from_last_surviving"]) for d in deaths)
    open_buckets = Counter(
        _step_bucket(d["steps_to_death_from_last_open_surviving"]) for d in deaths
    )

    surv_steps = [
        int(d["steps_to_death_from_last_surviving"])
        for d in deaths
        if d["steps_to_death_from_last_surviving"] is not None
    ]
    surv_frames_d = [
        int(d["frames_before_fatal_last_surviving"])
        for d in deaths
        if d["frames_before_fatal_last_surviving"] is not None
    ]
    open_steps = [
        int(d["steps_to_death_from_last_open_surviving"])
        for d in deaths
        if d["steps_to_death_from_last_open_surviving"] is not None
    ]
    open_frames_d = [
        int(d["frames_before_fatal_last_open_surviving"])
        for d in deaths
        if d["frames_before_fatal_last_open_surviving"] is not None
    ]
    trap_runs = [int(d["v8_trap_run"]) for d in deaths]
    lengths = [int(d["length"]) for d in deaths]

    def _opp_within_table(key_field: str, sub_key: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        denom = sum(1 for d in deaths if d.get(key_field) is not None)
        for k in K_STEPS:
            ks = str(k)
            c = sum(
                1
                for d in deaths
                if d.get(key_field) is not None and bool(d[key_field][sub_key].get(ks, False))
            )
            out[ks] = {
                "count": c,
                "denominator": denom,
                "fraction": round(c / denom, 4) if denom > 0 else 0.0,
            }
        return out

    # Per-world clustering stats
    wc_arr = np.asarray(world_counts, dtype=np.float64)
    wc_mean = float(np.mean(wc_arr)) if len(wc_arr) else 0.0
    wc_var = float(np.var(wc_arr, ddof=1)) if len(wc_arr) > 1 else 0.0
    wc_hist = Counter(int(c) for c in world_counts)

    # Shadow v8 metrics
    v8_diff_k = {}
    for k in range(1, 11):
        ks = str(k)
        c = sum(1 for d in deaths if bool(d["v8_differs_within_k"].get(ks, False)))
        v8_diff_k[ks] = {
            "count": c,
            "fraction": round(c / n, 4) if n > 0 else 0.0,
        }
    v8_surv_last = sum(1 for d in deaths if bool(d["v8_survives_at_last_surviving"]))
    v8_surv_open = sum(1 for d in deaths if bool(d["v8_survives_at_last_open_surviving"]))

    # Premise tests (2-8 step opponent pinches)
    # 1. Exact H-to-death steps in 2..8
    in_2_8_surv = [
        d
        for d in deaths
        if d["steps_to_death_from_last_surviving"] is not None
        and 2 <= int(d["steps_to_death_from_last_surviving"]) <= 8
    ]
    in_2_8_open = [
        d
        for d in deaths
        if d["steps_to_death_from_last_open_surviving"] is not None
        and 2 <= int(d["steps_to_death_from_last_open_surviving"]) <= 8
    ]

    opp_cause_set = {"opponent head-on", "opponent body"}
    strict_pinch_surv = [d for d in in_2_8_surv if d["cause_class"] in opp_cause_set]
    strict_pinch_open = [d for d in in_2_8_open if d["cause_class"] in opp_cause_set]

    prox_pinch_surv = [
        d
        for d in in_2_8_surv
        if d["proximity_at_last_surviving"] is not None
        and bool(d["proximity_at_last_surviving"]["opp_body_within_k"]["8"])
        and (
            d["cause_class"] in opp_cause_set
            or (
                d["pocket_boundary_at_last_surviving"] is not None
                and d["pocket_boundary_at_last_surviving"]["dominant_snake_blocker"] == "opponent"
            )
        )
    ]
    prox_pinch_open = [
        d
        for d in in_2_8_open
        if d["proximity_at_last_open_surviving"] is not None
        and bool(d["proximity_at_last_open_surviving"]["opp_body_within_k"]["8"])
        and (
            d["cause_class"] in opp_cause_set
            or (
                d["pocket_boundary_at_last_open_surviving"] is not None
                and d["pocket_boundary_at_last_open_surviving"]["dominant_snake_blocker"]
                == "opponent"
            )
        )
    ]
    opp_head_within8_in_2_8_surv = [
        d
        for d in in_2_8_surv
        if d["proximity_at_last_surviving"] is not None
        and bool(d["proximity_at_last_surviving"]["opp_head_within_k"]["8"])
    ]
    opp_head_within8_in_2_8_open = [
        d
        for d in in_2_8_open
        if d["proximity_at_last_open_surviving"] is not None
        and bool(d["proximity_at_last_open_surviving"]["opp_head_within_k"]["8"])
    ]
    any_opp_within8_in_2_8_surv = [
        d for d in in_2_8_surv if bool(d["proximity_at_death"]["opp_body_within_k"]["8"])
    ]
    any_opp_within8_in_2_8_open = [
        d for d in in_2_8_open if bool(d["proximity_at_death"]["opp_body_within_k"]["8"])
    ]

    # Cross-tab cause_class x step_bucket (for both surv and open)
    cross_surv: dict[str, dict[str, int]] = {
        c: {b: 0 for b in STEP_BUCKET_ORDER} for c in CAUSE_CLASSES
    }
    cross_open: dict[str, dict[str, int]] = {
        c: {b: 0 for b in STEP_BUCKET_ORDER} for c in CAUSE_CLASSES
    }
    for d in deaths:
        c = d["cause_class"]
        bs = _step_bucket(d["steps_to_death_from_last_surviving"])
        bo = _step_bucket(d["steps_to_death_from_last_open_surviving"])
        cross_surv[c][bs] += 1
        cross_open[c][bo] += 1

    return {
        "total_deaths": n,
        "alive_frames": alive_frames,
        "deaths_per_1k_alive_frames": (
            round(1000.0 * n / alive_frames, 4) if alive_frames > 0 else 0.0
        ),
        "length_quantiles": _quantiles(lengths),
        "cause_class_distribution": {
            c: {
                "count": int(cause_counts.get(c, 0)),
                "fraction": round(cause_counts.get(c, 0) / n, 4) if n > 0 else 0.0,
            }
            for c in CAUSE_CLASSES
        },
        "topological_cause_class_distribution": {
            c: {
                "count": int(topo_cause_counts.get(c, 0)),
                "fraction": round(topo_cause_counts.get(c, 0) / n, 4) if n > 0 else 0.0,
            }
            for c in CAUSE_CLASSES
        },
        "steps_to_death_from_last_surviving_distribution": {
            b: {
                "count": int(surv_buckets.get(b, 0)),
                "fraction": round(surv_buckets.get(b, 0) / n, 4) if n > 0 else 0.0,
            }
            for b in STEP_BUCKET_ORDER
        },
        "steps_to_death_from_last_open_surviving_distribution": {
            b: {
                "count": int(open_buckets.get(b, 0)),
                "fraction": round(open_buckets.get(b, 0) / n, 4) if n > 0 else 0.0,
            }
            for b in STEP_BUCKET_ORDER
        },
        "steps_to_death_from_last_surviving_quantiles": _quantiles(surv_steps),
        "frames_before_fatal_last_surviving_quantiles": _quantiles(surv_frames_d),
        "steps_to_death_from_last_open_surviving_quantiles": _quantiles(open_steps),
        "frames_before_fatal_last_open_surviving_quantiles": _quantiles(open_frames_d),
        "v8_trap_run_quantiles": _quantiles(trap_runs),
        "enclosed_at_fatal_count": sum(1 for d in deaths if bool(d["fatal_enclosed"])),
        "trap_run_ge_3_count": sum(1 for d in deaths if int(d["v8_trap_run"]) >= 3),
        "cause_by_step_bucket_surviving": cross_surv,
        "cause_by_step_bucket_open_surviving": cross_open,
        "opponent_within_k": {
            "at_death_head_l1": _opp_within_table("proximity_at_death", "opp_head_within_k"),
            "at_death_body_l1": _opp_within_table("proximity_at_death", "opp_body_within_k"),
            "at_last_surviving_head_l1": _opp_within_table(
                "proximity_at_last_surviving", "opp_head_within_k"
            ),
            "at_last_surviving_body_l1": _opp_within_table(
                "proximity_at_last_surviving", "opp_body_within_k"
            ),
            "at_last_open_surviving_head_l1": _opp_within_table(
                "proximity_at_last_open_surviving", "opp_head_within_k"
            ),
            "at_last_open_surviving_body_l1": _opp_within_table(
                "proximity_at_last_open_surviving", "opp_body_within_k"
            ),
        },
        "per_world_clustering": {
            "worlds": len(world_counts),
            "worlds_with_zero_deaths": int(sum(1 for c in world_counts if c == 0)),
            "worlds_with_ge_1_death": int(sum(1 for c in world_counts if c >= 1)),
            "mean_per_world": round(wc_mean, 4),
            "std_per_world": round(math.sqrt(wc_var), 4),
            "var_per_world": round(wc_var, 4),
            "dispersion_var_over_mean": (round(wc_var / wc_mean, 4) if wc_mean > 0 else None),
            "min_per_world": int(min(world_counts)) if world_counts else 0,
            "median_per_world": (round(float(np.median(wc_arr)), 3) if len(wc_arr) else 0.0),
            "max_per_world": int(max(world_counts)) if world_counts else 0,
            "count_histogram": {str(k): int(v) for k, v in sorted(wc_hist.items())},
        },
        "shadow_v8_audit": {
            "differs_within_k_frames": v8_diff_k,
            "v8_action_survives_at_last_surviving_count": v8_surv_last,
            "v8_action_survives_at_last_surviving_fraction": (
                round(v8_surv_last / n, 4) if n > 0 else 0.0
            ),
            "v8_action_survives_at_last_open_surviving_count": v8_surv_open,
            "v8_action_survives_at_last_open_surviving_fraction": (
                round(v8_surv_open / n, 4) if n > 0 else 0.0
            ),
        },
        "opponent_pinch_2_to_8_step_tests": {
            "all_2_to_8_step_surviving": {
                "count": len(in_2_8_surv),
                "fraction_of_regime_deaths": round(len(in_2_8_surv) / n, 4) if n > 0 else 0.0,
            },
            "all_2_to_8_step_open_surviving": {
                "count": len(in_2_8_open),
                "fraction_of_regime_deaths": round(len(in_2_8_open) / n, 4) if n > 0 else 0.0,
            },
            "strict_opponent_collision_and_2_to_8_step_surviving": {
                "count": len(strict_pinch_surv),
                "fraction_of_regime_deaths": (
                    round(len(strict_pinch_surv) / n, 4) if n > 0 else 0.0
                ),
            },
            "strict_opponent_collision_and_2_to_8_step_open_surviving": {
                "count": len(strict_pinch_open),
                "fraction_of_regime_deaths": (
                    round(len(strict_pinch_open) / n, 4) if n > 0 else 0.0
                ),
            },
            "opponent_dominant_or_collision_within8_and_2_to_8_step_surviving": {
                "count": len(prox_pinch_surv),
                "fraction_of_regime_deaths": (round(len(prox_pinch_surv) / n, 4) if n > 0 else 0.0),
            },
            "opponent_dominant_or_collision_within8_and_2_to_8_step_open_surviving": {
                "count": len(prox_pinch_open),
                "fraction_of_regime_deaths": (round(len(prox_pinch_open) / n, 4) if n > 0 else 0.0),
            },
            "opponent_head_within8_and_2_to_8_step_surviving": {
                "count": len(opp_head_within8_in_2_8_surv),
                "fraction_of_regime_deaths": (
                    round(len(opp_head_within8_in_2_8_surv) / n, 4) if n > 0 else 0.0
                ),
            },
            "opponent_head_within8_and_2_to_8_step_open_surviving": {
                "count": len(opp_head_within8_in_2_8_open),
                "fraction_of_regime_deaths": (
                    round(len(opp_head_within8_in_2_8_open) / n, 4) if n > 0 else 0.0
                ),
            },
            "any_opponent_body_within8_and_2_to_8_step_surviving": {
                "count": len(any_opp_within8_in_2_8_surv),
                "fraction_of_regime_deaths": (
                    round(len(any_opp_within8_in_2_8_surv) / n, 4) if n > 0 else 0.0
                ),
            },
            "any_opponent_body_within8_and_2_to_8_step_open_surviving": {
                "count": len(any_opp_within8_in_2_8_open),
                "fraction_of_regime_deaths": (
                    round(len(any_opp_within8_in_2_8_open) / n, 4) if n > 0 else 0.0
                ),
            },
        },
    }


def aggregate_death_audit(
    world_results: Sequence[dict[str, Any]],
    wall_seconds: float,
    workers: int,
) -> dict[str, Any]:
    """Aggregate per-world death audit results into full Stage 0b summary tables."""
    ordered_worlds = sorted(world_results, key=lambda w: int(w["world_index"]))
    all_deaths = [d for w in ordered_worlds for d in w["deaths"]]
    l_lt_600_deaths = [d for d in all_deaths if int(d["length"]) < 600]
    l_ge_600_deaths = [d for d in all_deaths if int(d["length"]) >= 600]

    total_l_lt_600_alive = sum(int(w["l_lt_600_alive_frames"]) for w in ordered_worlds)
    total_l_ge_600_alive = sum(int(w["l_ge_600_alive_frames"]) for w in ordered_worlds)
    total_alive = total_l_lt_600_alive + total_l_ge_600_alive
    total_dead_frames = sum(int(w["dead_frames"]) for w in ordered_worlds)

    world_counts_all = [int(w["total_deaths"]) for w in ordered_worlds]
    world_counts_lt600 = [int(w["l_lt_600_deaths"]) for w in ordered_worlds]
    world_counts_ge600 = [int(w["l_ge_600_deaths"]) for w in ordered_worlds]

    summary_lt600 = _summarize_regime(l_lt_600_deaths, world_counts_lt600, total_l_lt_600_alive)
    summary_ge600 = _summarize_regime(l_ge_600_deaths, world_counts_ge600, total_l_ge_600_alive)
    summary_all = _summarize_regime(all_deaths, world_counts_all, total_alive)

    # Evaluate whether the "2-8-step opponent pinches cause most (>50%) L<600 deaths" premise holds
    pinch_tests = summary_lt600["opponent_pinch_2_to_8_step_tests"]
    strict_surv_frac = pinch_tests["strict_opponent_collision_and_2_to_8_step_surviving"][
        "fraction_of_regime_deaths"
    ]
    strict_open_frac = pinch_tests["strict_opponent_collision_and_2_to_8_step_open_surviving"][
        "fraction_of_regime_deaths"
    ]
    broad_surv_frac = pinch_tests[
        "opponent_dominant_or_collision_within8_and_2_to_8_step_surviving"
    ]["fraction_of_regime_deaths"]
    broad_open_frac = pinch_tests[
        "opponent_dominant_or_collision_within8_and_2_to_8_step_open_surviving"
    ]["fraction_of_regime_deaths"]
    head8_surv_frac = pinch_tests["opponent_head_within8_and_2_to_8_step_surviving"][
        "fraction_of_regime_deaths"
    ]
    head8_open_frac = pinch_tests["opponent_head_within8_and_2_to_8_step_open_surviving"][
        "fraction_of_regime_deaths"
    ]
    all_2_8_surv_frac = pinch_tests["all_2_to_8_step_surviving"]["fraction_of_regime_deaths"]
    all_2_8_open_frac = pinch_tests["all_2_to_8_step_open_surviving"]["fraction_of_regime_deaths"]

    premise_holds = bool(
        strict_surv_frac > 0.5
        or strict_open_frac > 0.5
        or broad_surv_frac > 0.5
        or broad_open_frac > 0.5
        or head8_surv_frac > 0.5
        or head8_open_frac > 0.5
    )

    per_world_summary = [
        {
            "world_index": int(w["world_index"]),
            "seed": int(w["seed"]),
            "mean_mass": w["mean_mass"],
            "l_lt_600_alive_frames": int(w["l_lt_600_alive_frames"]),
            "l_ge_600_alive_frames": int(w["l_ge_600_alive_frames"]),
            "dead_frames": int(w["dead_frames"]),
            "total_deaths": int(w["total_deaths"]),
            "l_lt_600_deaths": int(w["l_lt_600_deaths"]),
            "l_ge_600_deaths": int(w["l_ge_600_deaths"]),
            "wall_seconds": w["wall_seconds"],
        }
        for w in ordered_worlds
    ]

    # Compact per-death table (without the bulky lookback_records dict) for fast inspection
    compact_deaths = [{k: v for k, v in d.items() if k != "lookback_records"} for d in all_deaths]

    return {
        "schema": "t34-hero-stage0b-death-audit/v1",
        "namespace": NAMESPACE,
        "world_count": len(ordered_worlds),
        "horizon_per_world": HORIZON,
        "total_frames_simulated": len(ordered_worlds) * HORIZON,
        "total_mirror_identity_frames": sum(
            int(w["mirror_identity_frames"]) for w in ordered_worlds
        ),
        "workers": workers,
        "wall_seconds": round(wall_seconds, 3),
        "sum_worker_wall_seconds": round(sum(float(w["wall_seconds"]) for w in ordered_worlds), 3),
        "overall_mean_mass": (
            round(float(np.mean([w["mean_mass"] for w in ordered_worlds])), 4)
            if ordered_worlds
            else 0.0
        ),
        "frame_exposure": {
            "l_lt_600_alive_frames": total_l_lt_600_alive,
            "l_ge_600_alive_frames": total_l_ge_600_alive,
            "total_alive_frames": total_alive,
            "dead_frames": total_dead_frames,
        },
        "total_deaths": len(all_deaths),
        "l_lt_600_deaths": len(l_lt_600_deaths),
        "l_ge_600_deaths": len(l_ge_600_deaths),
        "premise_evaluation": {
            "premise": "2-8-step opponent pinches cause most (>50%) L<600 deaths",
            "premise_holds": premise_holds,
            "l_lt_600_total_deaths": len(l_lt_600_deaths),
            "strict_opponent_collision_2_to_8_step_surviving_fraction": strict_surv_frac,
            "strict_opponent_collision_2_to_8_step_open_surviving_fraction": strict_open_frac,
            "broad_opponent_dominant_or_collision_2_to_8_step_surviving_fraction": (
                broad_surv_frac
            ),
            "broad_opponent_dominant_or_collision_2_to_8_step_open_surviving_fraction": (
                broad_open_frac
            ),
            "opponent_head_within8_2_to_8_step_surviving_fraction": head8_surv_frac,
            "opponent_head_within8_2_to_8_step_open_surviving_fraction": head8_open_frac,
            "all_causes_2_to_8_step_surviving_fraction": all_2_8_surv_frac,
            "all_causes_2_to_8_step_open_surviving_fraction": all_2_8_open_frac,
        },
        "regimes": {
            "L<600": summary_lt600,
            "L>=600": summary_ge600,
            "ALL": summary_all,
        },
        "per_world_summary": per_world_summary,
        "deaths": compact_deaths,
    }


def run_stage0b_audit(
    *,
    count: int = COUNT,
    horizon: int = HORIZON,
    workers: int = 8,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Run the 48-world Stage 0b bare B8 death-cause audit with at most 8 workers."""
    if workers < 1 or workers > 8:
        raise ValueError("workers must be in 1..8")
    _enforce_single_thread_env()
    alloc = allocation(NAMESPACE, count)
    worlds = list(alloc["worlds"])

    out_dir.mkdir(parents=True, exist_ok=True)
    worlds_dir = out_dir / "worlds"
    worlds_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []

    if workers == 1:
        for row in worlds:
            res = run_world_death_audit(row, horizon=horizon, worlds_dir=worlds_dir)
            results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fut_to_row = {
                pool.submit(run_world_death_audit, row, horizon, worlds_dir): row for row in worlds
            }
            for fut in as_completed(fut_to_row):
                res = fut.result()
                results.append(res)

    wall_seconds = time.perf_counter() - t0
    summary_path = out_dir / "DEATH_AUDIT.json"
    if summary_path.is_file():
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and float(prev.get("wall_seconds", 0.0)) > wall_seconds:
                wall_seconds = float(prev["wall_seconds"])
        except Exception:
            pass
    summary = aggregate_death_audit(results, wall_seconds=wall_seconds, workers=workers)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_stage0b_audit(
        count=args.count,
        horizon=args.horizon,
        workers=args.workers,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                "world_count": summary["world_count"],
                "total_deaths": summary["total_deaths"],
                "l_lt_600_deaths": summary["l_lt_600_deaths"],
                "l_ge_600_deaths": summary["l_ge_600_deaths"],
                "wall_seconds": summary["wall_seconds"],
                "premise_evaluation": summary["premise_evaluation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
