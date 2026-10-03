#!/usr/bin/env python3
"""DEVELOPMENT DIAGNOSTIC (Tier-0/1): death census of the Apex champion + released v5 veto.

Not a screen, not gate evidence, no candidate and no decision rule. It records every
hero death of the released Watch configuration (champion + ``free-space-veto/v5-boost-
aware``) on the LIVE engine and asks what killed it, to pick the next intervention.
v5 is not ported to the SIMD engine, so this is ``tournament_eval.rollout`` with the v5
hero installer (``dev_screen.hero_veto_installer`` + ``install_boost_aware_veto``, the v5
screen's arm B), at H5000, profile ``promotion-v2-watch-rect``, strict balanced rosters,
all three mixes, on the fresh namespace :data:`DOMAIN`.

Capture (read-only; nothing the hero sees or does changes):

* **Decisions.** The installed v5 veto instance's ``apply`` is wrapped (original first,
  its action returned unchanged). Each hero decision snapshots, in cell units, the hero
  body (ordered), length, heading, boost counter, every other live snake's cells, the
  food cells the hero was given, v2's one-step counts/need (recomputed from the same
  ``_get_free_space_features`` call v5 makes), the masked Q row, the action mask, the
  base and final actions and v5's decision reason (from the counter deltas). A ring of
  :data:`WINDOW` decisions per episode bounds memory.
* **Post-move world.** ``GameLogic.check_collisions`` is wrapped for the duration of an
  episode: while the hero is alive it keeps (only) the latest post-move, pre-collision
  state of every snake (cells, traversed head cells, heading, logical length) plus the
  food at that point. At a death this is the fatal frame.

Analysis (offline, pure):

* **Self deaths**: the trap-horizon backward point-of-no-return search of
  ``diagnose.py`` (unchanged: exact own moves, other snakes static walls, count-or-depth
  primary criterion with depth-only and count-only sensitivities, node budget, unknown
  never guessed), plus v5's view at the PNR (reason, offline two-cell landing counts) and
  a category (boost / normal-speed fatal choice, taken action escaped, enclosed early).
* **Every death** (self, head_on, enemy_body, wall): an exact one-frame counterfactual at
  the fatal decision. Other snakes act on pre-move snapshots, so their moves do not
  depend on the hero's action; each hero action is replayed against the captured
  post-move world with the live collision rules (wall, self, head-on incl. path swap and
  v2 size resolution, enemy body). It reports which masked-legal alternatives avoided
  death that frame, and whether each such alternative then has a static-world escape
  (the same count-or-depth search, others frozen at their post-move cells).

Usage (CPU slot 1, AC power required; output directory is create-only):
  SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=2 ./venv/bin/python \\
      research/trap_horizon_20261001/diagnose_live_v5.py
Smoke (separate smoke namespace, short horizon, scratch output):
  ... diagnose_live_v5.py --smoke --out /tmp/x --worlds-per-mix 1 --frames 400 --mixes frozen
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
REQUIRED_ENV = {"SNAKE_DQN_DEVICE": "cpu", "OMP_NUM_THREADS": "2"}

import argparse  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import (  # noqa: E402
    Any,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.trap_horizon_20261001 import diagnose as base  # noqa: E402

SCHEMA = "death-census-v5-live/v1"
AUTHORITY = base.AUTHORITY
DOMAIN = "trap-horizon-v5-dev-v1"
SMOKE_DOMAIN = "trap-horizon-v5-dev-smoke-v1"
NAMESPACE = base.NAMESPACE
WORLDS_PER_MIX = 16
WINDOW = base.WINDOW
DEPTH = base.DEPTH
NODE_BUDGET = base.NODE_BUDGET
ARTIFACTS = base.ARTIFACTS
DEFAULT_OUT = ARTIFACTS / "trap-horizon-v5-20261002" / "run-v1"
SLOT_LOCK = base.SLOT_LOCK
EXCLUSION_PREFIX = base.EXCLUSION_PREFIX
ENCLOSED_EARLY_FRAMES = 20  # count-only PNR beyond this = "enclosed long before"

# v5 decision reasons, as integer codes in the saved windows.
V5_REASONS = (
    "kept",
    "v2_rule",
    "landing_same_direction_normal",
    "landing_v2_rule",
    "landing_no_eligible",
)
# Every apex-veto-v5-* domain (and the purposes it used) plus the v2 trap-horizon study.
V5_DOMAINS: Dict[str, Tuple[str, ...]] = {
    "apex-veto-v5-screen-v1": ("worlds",),
    "apex-veto-v5-screen-smoke-v1": ("worlds",),
    "apex-veto-v5-strict-dev-v1": ("worlds",),
    "apex-veto-v5-strict-final-v1": ("worlds",),
    "apex-veto-v5-strict-serving-v1": ("worlds",),
    "apex-veto-v5-strict-smoke-v1": ("worlds",),
    "apex-veto-v5-web-serving-v1": ("watch", "play", "parity"),
    "apex-veto-v5-web-serving-smoke-v1": ("watch", "play", "parity"),
    "trap-horizon-dev-v1": ("worlds",),
    "trap-horizon-dev-smoke-v1": ("worlds",),
}

Cell = Tuple[int, int]


# ---------------------------------------------------------------------------
# Live capture
# ---------------------------------------------------------------------------
def _cells(points: Sequence[Sequence[float]], size: int) -> List[Cell]:
    return [(int(x // size), int(y // size)) for x, y in points]


def _cell_array(points: Sequence[Sequence[float]], size: int) -> np.ndarray:
    return np.asarray(_cells(points, size), dtype=np.int16).reshape(-1, 2)


def _as_numpy(values: Any) -> np.ndarray:
    detach = getattr(values, "detach", None)
    if callable(detach):
        return detach().cpu().numpy()
    return np.asarray(values)


def _direction_index(direction: Sequence[int]) -> int:
    return base.CARDINAL.index((int(direction[0]), int(direction[1])))


class LiveRecorder:
    """Per-episode ring of hero decisions plus the latest post-move world (bounded)."""

    def __init__(self, window: int = WINDOW) -> None:
        self.window = int(window)
        self.reset()
        self.world_meta: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=self.window)
        self.decisions = 0
        self.hero: Any = None
        self.veto: Any = None
        self.game_state: Any = None
        self.food: Sequence[Any] = ()
        self.post: Optional[Dict[str, Any]] = None

    # -- hooks --------------------------------------------------------------
    def attach(self, hero: Any, veto: Any) -> None:
        """Wrap ``hero.update`` (stash the food it is given) and ``veto.apply`` (capture)."""
        from src.core.game_config import GameConfig
        from src.evaluation.safety_veto_v3 import grid_for

        self.hero, self.veto = hero, veto
        grid = grid_for(hero)
        if grid.circular:
            raise ValueError("the census assumes a rectangular arena")
        self.world_meta = {
            "grid_width": int(grid.width),
            "grid_height": int(grid.height),
            "min_boost_length": int(GameConfig.MIN_BOOST_LENGTH),
            "boost_cost_frames": int(GameConfig.BOOST_LENGTH_COST_FRAMES),
            "trail_food": int(GameConfig.MECHANICS_VERSION) == 2,
            "segment_size": int(grid.segment_size),
            "mechanics_version": int(GameConfig.MECHANICS_VERSION),
        }
        original_update = hero.update
        original_apply = veto.apply

        def update(other_snakes, food, **kwargs):  # type: ignore[no-untyped-def]
            self.food = food
            return original_update(other_snakes, food, **kwargs)

        def apply(snake, other_snakes, q_values, action_mask, base_action):  # type: ignore
            before = (veto.boost.base_landing_failed, veto.boost.boost_landing_no_eligible)
            before += (veto.boost.boost_to_normal_same_direction, veto.boost.kept)
            action = original_apply(snake, other_snakes, q_values, action_mask, base_action)
            self.capture(snake, other_snakes, q_values, action_mask, base_action, action, before)
            return action

        hero.update = update
        veto.apply = apply

    def capture(
        self,
        snake: Any,
        other_snakes: Sequence[Any],
        q_values: Any,
        action_mask: Any,
        base_action: int,
        action: int,
        before: Tuple[int, int, int, int],
    ) -> None:
        from src.evaluation.safety_veto import free_space_threshold

        size = int(self.world_meta["segment_size"])  # type: ignore[index]
        others = [o for o in other_snakes if o is not snake and o.is_alive]
        features = snake._get_free_space_features(list(other_snakes))
        cap, need = free_space_threshold(snake.length, snake._logical_length())
        boost = self.veto.boost
        failed, no_eligible, same_dir, kept = before
        if boost.base_landing_failed > failed:
            if boost.boost_landing_no_eligible > no_eligible:
                reason = "landing_no_eligible"
            elif boost.boost_to_normal_same_direction > same_dir:
                reason = "landing_same_direction_normal"
            else:
                reason = "landing_v2_rule"
        else:
            reason = "kept" if boost.kept > kept else "v2_rule"
        others_cells = [_cell_array(o.segments, size) for o in others]
        q = np.asarray(_as_numpy(q_values), dtype=np.float32).reshape(-1)
        mask = np.asarray(_as_numpy(action_mask), dtype=bool).reshape(-1)
        self.buffer.append(
            {
                "frame": int(self.game_state.frame) if self.game_state is not None else -1,
                "body": _cell_array(snake.segments, size),
                "length": int(snake.length),
                "direction": _direction_index(snake.direction),
                "boost_frames": int(snake.boost_frames),
                "others": (
                    np.concatenate(others_cells) if others_cells else np.zeros((0, 2), np.int16)
                ),
                "food": _cell_array(list(self.food), size),
                "v2_counts": np.asarray(
                    [int(round(float(f) * cap)) for f in features], dtype=np.int64
                ),
                "v2_need": int(need),
                "base": int(base_action),
                "final": int(action),
                "q": q,
                "mask": mask,
                "v5_reason": V5_REASONS.index(reason),
            }
        )
        self.decisions += 1

    def observe_collisions(self, snakes: Sequence[Any]) -> None:
        """Keep the post-move, pre-collision world while the hero is alive."""
        hero = self.hero
        if hero is None or not hero.is_alive or not any(s is hero for s in snakes):
            return
        size = int(self.world_meta["segment_size"])  # type: ignore[index]
        rows = []
        for snake in snakes:
            moved = list(getattr(snake, "last_move_positions", None) or [])
            rows.append(
                {
                    "is_hero": snake is hero,
                    "id": int(snake.id),
                    "alive": bool(snake.is_alive),
                    "cells": _cells(snake.segments, size),
                    "traversed": _cells(moved, size) if moved else _cells([snake.head], size),
                    "direction": _direction_index(snake.direction),
                    "logical_length": int(snake._logical_length()),
                }
            )
        gs = self.game_state
        self.post = {
            "frame": int(gs.frame) if gs is not None else -1,
            "snakes": rows,
            "food": _cells(list(gs.food) if gs is not None else [], size),
        }


@contextmanager
def live_capture(recorder: LiveRecorder) -> Iterator[LiveRecorder]:
    """Route rollout's hero veto install to v5 + capture; observe the game state/collisions.

    Patches (restored on exit): ``tournament_eval.create_training_game_state`` (to keep
    a reference to the episode's ``GameState``), ``GameLogic.check_collisions`` (to keep
    the post-move world; the original is called and its result returned unchanged) and,
    through ``dev_screen.hero_veto_installer``, rollout's hero veto install (the original
    vector61 guard still runs first, then v5 is installed exactly as the v5 screen's arm B).
    """
    from research.apex_safety_20260926 import dev_screen as ds
    from src.evaluation.safety_veto_v5 import install_boost_aware_veto
    from src.game.game_logic import GameLogic
    from src.scripts import tournament_eval as te

    original_create = te.create_training_game_state
    original_check = GameLogic.__dict__["check_collisions"]

    def create(*args, **kwargs):  # type: ignore[no-untyped-def]
        gs = original_create(*args, **kwargs)
        recorder.game_state = gs
        return gs

    def check(snakes):  # type: ignore[no-untyped-def]
        recorder.observe_collisions(snakes)
        return original_check.__func__(snakes)

    def install(hero: Any) -> Any:
        veto = install_boost_aware_veto(hero)
        recorder.attach(hero, veto)
        return veto

    te.create_training_game_state = create
    GameLogic.check_collisions = staticmethod(check)
    try:
        with ds.hero_veto_installer(install) as installed:
            recorder.installed = installed  # type: ignore[attr-defined]
            yield recorder
    finally:
        te.create_training_game_state = original_create
        GameLogic.check_collisions = original_check


# ---------------------------------------------------------------------------
# Window packing (diagnose.pack_window plus v5 reasons)
# ---------------------------------------------------------------------------
def pack_live_window(snaps: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    out = base.pack_window(snaps)
    out["v5_reason"] = np.asarray([int(s["v5_reason"]) for s in snaps], dtype=np.int64)
    return out


def unpack_live_window(packed: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    snaps = base.unpack_window(packed)
    for snap, code in zip(snaps, packed["v5_reason"]):
        snap["v5_reason"] = int(code)
    return snaps


# ---------------------------------------------------------------------------
# One-frame counterfactual against the post-move world (pure)
# ---------------------------------------------------------------------------
def traversed_cells(state: base.HeroState, child: base.HeroState, boosted: bool) -> List[Cell]:
    """Head cells entered this frame (first, then second for a boost)."""
    return [child.body[1], child.body[0]] if boosted else [child.body[0]]


def _paths_crossed(path_a: Sequence[Cell], path_b: Sequence[Cell]) -> bool:
    """``GameLogic._head_paths_crossed`` on the cell lattice (swap along any sub-step)."""
    for start_a, end_a in zip(path_a, path_a[1:]):
        for start_b, end_b in zip(path_b, path_b[1:]):
            if start_a == end_b and start_b == end_a:
                return True
    return False


def one_frame_outcome(
    meta: Mapping[str, Any],
    state: base.HeroState,
    action: int,
    others: Sequence[Mapping[str, Any]],
    food: frozenset,
    headon_ratio: float,
) -> Dict[str, Any]:
    """The hero's fate this frame for ``action`` given the other snakes' actual moves.

    ``others`` are the post-move, pre-collision rows (``cells`` head first, ``traversed``,
    ``direction``, ``logical_length``, ``alive``) of every other snake. Order of checks
    follows ``GameLogic.check_collisions`` for the hero (slot 0, processed first): wall,
    self, then per other snake head (shared traversed cell or swapped head paths; v2
    size resolution) else body (a traversed cell in its ``cells[1:]``).
    """
    world = base.StaticWorld(
        width=int(meta["grid_width"]),
        height=int(meta["grid_height"]),
        blocked=frozenset(),
        food=frozenset(food),
        min_boost_length=int(meta["min_boost_length"]),
        boost_cost_frames=int(meta["boost_cost_frames"]),
        trail_food=bool(meta["trail_food"]),
    )
    child, cause = base.step_hero(world, state, int(action))
    if child is None:
        return {"cause": cause, "child": None, "against": None}
    boosted = int(action) >= base.NUM_DIRECTIONS and state.length >= world.min_boost_length
    moved = traversed_cells(state, child, boosted)
    hero_path = [state.body[0]] + moved
    mechanics_v2 = int(meta.get("mechanics_version", 2)) == 2
    for index, other in enumerate(others):
        if not other["alive"]:
            continue
        o_moved = [tuple(c) for c in other["traversed"]]
        dx, dy = base.CARDINAL[int(other["direction"])]
        first = o_moved[0]
        o_path = [(first[0] - dx, first[1] - dy)] + o_moved
        if set(moved) & set(o_moved) or _paths_crossed(hero_path, o_path):
            if mechanics_v2 and child.length >= headon_ratio * int(other["logical_length"]):
                continue  # the hero wins this head-on (kill)
            return {"cause": "head_on", "child": child, "against": index}
        body = {tuple(c) for c in list(other["cells"])[1:]}
        if any(c in body for c in moved):
            return {"cause": "enemy_body", "child": child, "against": index}
    return {"cause": "alive", "child": child, "against": None}


def escape_status(
    world: base.StaticWorld, state: base.HeroState, depth: int, budget: int
) -> Tuple[str, int]:
    """Count-or-depth escape from a state reached after one frame (others static)."""
    search = base.EscapeSearch(world, depth, budget, base.TAIL_SLACK, base.COUNT_OR_DEPTH)
    search.nodes = 0
    try:
        found = search._visit(state, 1)
    except base.BudgetExhausted:
        return base.UNKNOWN, search.nodes
    return (base.ESCAPE if found else base.NO_ESCAPE), search.nodes


def fatal_frame_analysis(
    snap: Mapping[str, Any],
    post: Mapping[str, Any],
    meta: Mapping[str, Any],
    actual_cause: Optional[str],
    depth: int = DEPTH,
    budget: int = NODE_BUDGET,
    headon_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Every action at the fatal decision against the actual post-move world."""
    if headon_ratio is None:
        from src.core.mechanics_constants import HEADON_SIZE_RATIO

        headon_ratio = float(HEADON_SIZE_RATIO)
    _, state = base.world_and_state(snap, meta)
    others = [row for row in post["snakes"] if not row["is_hero"]]
    pre_food = {(int(x), int(y)) for x, y in snap["food"]}
    food = frozenset(pre_food | {tuple(c) for c in post["food"]})
    pre_others = {(int(x), int(y)) for x, y in snap["others"]}
    post_blocked = frozenset(tuple(c) for row in others if row["alive"] for c in row["cells"])
    escape_world = base.StaticWorld(
        width=int(meta["grid_width"]),
        height=int(meta["grid_height"]),
        blocked=post_blocked,
        food=food,
        min_boost_length=int(meta["min_boost_length"]),
        boost_cost_frames=int(meta["boost_cost_frames"]),
        trail_food=bool(meta["trail_food"]),
    )
    mask = [bool(x) for x in snap["mask"]]
    taken = int(snap["final"])
    actions: Dict[str, Any] = {}
    for action in range(base.NUM_ACTIONS):
        out = one_frame_outcome(meta, state, action, others, food, headon_ratio)
        row: Dict[str, Any] = {"legal": mask[action], "cause": out["cause"]}
        child = out["child"]
        if out["cause"] in ("head_on", "enemy_body") and child is not None:
            boosted = action >= base.NUM_DIRECTIONS and state.length >= int(
                meta["min_boost_length"]
            )
            hit = traversed_cells(state, child, boosted)
            row["hit_cells_static_pre_move"] = any(c in pre_others for c in hit)
        if out["cause"] == "alive" and child is not None:
            t0 = time.perf_counter()
            status, nodes = escape_status(escape_world, child, depth, budget)
            row.update({"escape": status, "nodes": nodes, "seconds": time.perf_counter() - t0})
        actions[str(action)] = row
    avoid = [a for a in range(base.NUM_ACTIONS) if actions[str(a)]["cause"] == "alive"]
    legal_avoid = [a for a in avoid if mask[a] and a != taken]
    legal_avoid_escape = [a for a in legal_avoid if actions[str(a)].get("escape") == base.ESCAPE]
    taken_row = actions[str(taken)]
    return {
        "taken": taken,
        "mask": mask,
        "actions": actions,
        "taken_model_cause": taken_row["cause"],
        "model_reproduces_cause": taken_row["cause"] == actual_cause,
        "taken_hit_static_pre_move": taken_row.get("hit_cells_static_pre_move"),
        "legal_alternatives_avoiding": legal_avoid,
        "legal_alternatives_avoiding_with_escape": legal_avoid_escape,
        "any_legal_alternative_avoids": bool(legal_avoid),
        "any_legal_alternative_avoids_with_escape": bool(legal_avoid_escape),
        "post_frame": int(post["frame"]),
    }


# ---------------------------------------------------------------------------
# Self-death extras: v5 view and landing counts at the PNR (pure)
# ---------------------------------------------------------------------------
def static_landing_count(
    world: base.StaticWorld, state: base.HeroState, direction: int, cap: int
) -> Optional[int]:
    """v5's ``landing_count`` for boost ``direction`` on a snapshot (None if no boost fires).

    Same rule as ``safety_veto_v5.landing_count``: simulate the boost's movement
    (``simulate_move``, no collision check), block the post-move body except the new head
    plus the other snakes, and flood-fill from the landing cell with ``cap``.
    """
    from src.evaluation.safety_veto_v3 import tail_aware_reachable
    from src.evaluation.safety_veto_v5 import simulate_move

    rel = int(direction) % base.NUM_DIRECTIONS
    heading = (state.direction + (-1 if rel == 0 else (1 if rel == 2 else 0))) % 4
    move = simulate_move(
        state.body,
        base.CARDINAL[heading],
        state.length,
        state.boost_frames,
        True,
        1,
        world.min_boost_length,
        world.boost_cost_frames,
    )
    if not move.boosted:
        return None
    blocked = set(world.blocked) | {tuple(c) for c in move.segments[1:]}
    start = tuple(int(v) for v in move.segments[0])
    return tail_aware_reachable(start, 0, blocked, {}, world.in_bounds, int(cap))


def v5_view(snap: Mapping[str, Any], meta: Mapping[str, Any]) -> Dict[str, Any]:
    from src.evaluation.safety_veto import free_space_threshold

    view = base._v2_view(snap)
    world, state = base.world_and_state(snap, meta)
    cap, need = free_space_threshold(state.length, max(1, state.length))
    view["v5_reason"] = V5_REASONS[int(snap["v5_reason"])]
    view["landing_counts"] = [
        static_landing_count(world, state, d, cap) for d in range(base.NUM_DIRECTIONS)
    ]
    view["cap"] = int(cap)
    return view


def enclosed_early(death: Mapping[str, Any]) -> Optional[bool]:
    """Count-only PNR more than :data:`ENCLOSED_EARLY_FRAMES` frames before death.

    Uses the exact value, or the lower bound of an unknown / beyond-window walk (a
    lower bound above the threshold is enough). ``None`` when the bound is too low to
    decide (an unknown walk whose lower bound is within the threshold).
    """
    walk = death["sensitivity"][base.COUNT_ONLY]["walk"]
    if walk["status"] == "exact":
        return int(walk["frames_before_death"]) > ENCLOSED_EARLY_FRAMES
    lower = walk.get("frames_before_lower")
    if lower is not None and int(lower) > ENCLOSED_EARLY_FRAMES:
        return True
    return None


def classify_self_death(death: Mapping[str, Any]) -> str:
    """Primary-criterion category of one analyzed self death.

    ``boost_fatal_choice`` / ``normal_fatal_choice``: the action taken at the PNR had no
    escape while another action did (split by whether the taken action was a boost);
    ``taken_escaped``: the taken action itself had an escape at the PNR (the trap closed
    later); ``unresolved``: no PNR found or the taken action's status is unknown.
    :func:`enclosed_early` is reported beside it, not folded in.
    """
    walk = death["walk"]
    if walk["pnr_index"] is None or walk["taken_status"] in (None, base.UNKNOWN):
        return "unresolved"
    if walk["fatal_choice"]:
        return (
            "boost_fatal_choice"
            if int(walk["taken_action"]) >= base.NUM_DIRECTIONS
            else "normal_fatal_choice"
        )
    if walk["taken_status"] == base.ESCAPE:
        return "taken_escaped"
    return "unresolved"


def pnr_escape_kinds(death: Mapping[str, Any]) -> str:
    """Which one-sided searches found an escape at the primary PNR (as diagnose.py)."""
    row = death.get("pnr_row") or {}
    kinds = set()
    for criterion, kind in (
        (base.DEPTH_ONLY, "survive_depth"),
        (base.COUNT_ONLY, "spacious_count"),
    ):
        if any(r["status"] == base.ESCAPE for r in row.get(criterion, {}).values()):
            kinds.add(kind)
    return "+".join(sorted(kinds)) or "none"


def analyze_live_death(
    snaps: Sequence[Mapping[str, Any]],
    post: Mapping[str, Any],
    meta: Mapping[str, Any],
    cause: str,
    depth: int = DEPTH,
    budget: int = NODE_BUDGET,
) -> Dict[str, Any]:
    """One death: fatal-frame counterfactual for every cause, PNR walk for self deaths."""
    started = time.perf_counter()
    out: Dict[str, Any] = {
        "fatal": fatal_frame_analysis(snaps[-1], post, meta, cause, depth, budget),
        "fatal_v5": v5_view(snaps[-1], meta),
        "fatal_length": int(snaps[-1]["length"]),
        "fatal_frame": int(snaps[-1]["frame"]),
        "window_frames": len(snaps),
        "model_check": base.model_replay_check(snaps, meta),
    }
    if cause == "self":
        pnr = base.analyze_death(snaps, meta, depth, budget)
        pnr.pop("model_check", None)
        out.update(pnr)
        walk = out["walk"]
        out["pnr_v5"] = (
            v5_view(snaps[walk["pnr_index"]], meta) if walk["pnr_index"] is not None else None
        )
        if walk["pnr_index"] is not None and walk["taken_action"] is not None:
            taken = int(walk["taken_action"])
            row = out["pnr_row"].get(base.COUNT_OR_DEPTH, {})
            out["pnr_same_direction_normal_status"] = (
                row.get(str(taken % base.NUM_DIRECTIONS)) if taken >= base.NUM_DIRECTIONS else None
            )
            mask = [bool(x) for x in snaps[walk["pnr_index"]]["mask"]]
            out["pnr_escaping_masked_legal"] = [a for a in walk["escaping_actions"] if mask[a]]
        out["category"] = classify_self_death(out)
        out["enclosed_early"] = enclosed_early(out)
        out["pnr_escape_kinds"] = pnr_escape_kinds(out)
    out["seconds_total"] = time.perf_counter() - started
    return out


def analyze_live_death_file(args: Tuple[str, str, Mapping[str, Any], str, int, int]):
    """Pool worker: load one saved death (window npz + post JSON) and analyze it."""
    window_path, post_path, meta, cause, depth, budget = args
    with np.load(window_path) as data:
        packed = {k: data[k] for k in data.files}
    post = json.loads(Path(post_path).read_text())
    return analyze_live_death(unpack_live_window(packed), post, meta, cause, depth, budget)


# ---------------------------------------------------------------------------
# Summary (pure)
# ---------------------------------------------------------------------------
def _tally(values: Any) -> Dict[str, int]:
    return base._tally(values)


def summarize_census(
    episodes: Sequence[Mapping[str, Any]], deaths: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Outcome table, self-death PNR summary and fatal-frame alternatives by cause."""
    outcomes: Dict[str, Dict[str, int]] = {}
    mass: Dict[str, List[float]] = {}
    v5_totals: Dict[str, Dict[str, float]] = {}
    for ep in episodes:
        label = str(ep["death_cause"]) if float(ep["deaths"]) >= 1 else "survived"
        cell = outcomes.setdefault(ep["mix"], {})
        cell[label] = cell.get(label, 0) + 1
        mass.setdefault(ep["mix"], []).append(float(ep["mass_integral"]))
        totals = v5_totals.setdefault(ep["mix"], {})
        for key, value in (ep.get("v5_diagnostics") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key in ("apply_seconds_max", "mean_apply_seconds"):
                    continue
                totals[key] = totals.get(key, 0) + value
    all_totals: Dict[str, float] = {}
    for totals in v5_totals.values():
        for key, value in totals.items():
            all_totals[key] = all_totals.get(key, 0) + value
    pooled: Dict[str, int] = {}
    for cell in outcomes.values():
        for key, value in cell.items():
            pooled[key] = pooled.get(key, 0) + value

    self_rows = [d for d in deaths if d["cause"] == "self"]
    by_cause: Dict[str, Dict[str, Any]] = {}
    for cause in sorted({d["cause"] for d in deaths}):
        rows = [d for d in deaths if d["cause"] == cause]
        by_cause[cause] = {
            "deaths": len(rows),
            "model_reproduces_cause": sum(bool(d["fatal"]["model_reproduces_cause"]) for d in rows),
            "any_legal_alternative_avoids": sum(
                bool(d["fatal"]["any_legal_alternative_avoids"]) for d in rows
            ),
            "any_legal_alternative_avoids_with_escape": sum(
                bool(d["fatal"]["any_legal_alternative_avoids_with_escape"]) for d in rows
            ),
            "taken_hit_static_pre_move": _tally(
                d["fatal"]["taken_hit_static_pre_move"] for d in rows
            ),
            "taken_boost": sum(int(d["fatal"]["taken"]) >= base.NUM_DIRECTIONS for d in rows),
            "fatal_v5_reason": _tally(d["fatal_v5"]["v5_reason"] for d in rows),
            "fatal_v2_outcome": _tally(d["fatal_v5"]["outcome"] for d in rows),
            "by_mix": _tally(d["mix"] for d in rows),
            "fatal_lengths": sorted(int(d["fatal_length"]) for d in rows),
        }
    summary: Dict[str, Any] = {
        "episodes": len(episodes),
        "episode_outcomes_by_mix": dict(sorted(outcomes.items())),
        "episode_outcomes_pooled": dict(sorted(pooled.items())),
        "mean_mass_integral_by_mix": {m: float(np.mean(v)) for m, v in sorted(mass.items())},
        "v5_counters_by_mix": dict(sorted(v5_totals.items())),
        "v5_counters_total": dict(sorted(all_totals.items())),
        "fatal_frame_by_cause": by_cause,
        "model_check": {
            "transitions_compared": sum(d["model_check"]["transitions_compared"] for d in deaths),
            "mismatches": sum(d["model_check"]["mismatches"] for d in deaths),
            "fatal_action_static_model_outcome_self_deaths": _tally(
                d["model_check"]["fatal_action_model_outcome"] for d in self_rows
            ),
        },
    }
    if self_rows:
        found = [d for d in self_rows if d["walk"]["pnr_index"] is not None]
        summary["self"] = {
            "pnr": base.summarize_walks(self_rows),
            "pnr_sensitivity": {
                c: base.summarize_walks(
                    [{"mix": d["mix"], "walk": d["sensitivity"][c]["walk"]} for d in self_rows]
                )
                for c in (base.DEPTH_ONLY, base.COUNT_ONLY)
            },
            "categories": _tally(d["category"] for d in self_rows),
            "categories_by_mix": {
                mix: _tally(d["category"] for d in self_rows if d["mix"] == mix)
                for mix in sorted({d["mix"] for d in self_rows})
            },
            "enclosed_early": _tally(d["enclosed_early"] for d in self_rows),
            "category_by_enclosed_early": _tally(
                f"{d['category']}|enclosed_early={d['enclosed_early']}" for d in self_rows
            ),
            "pnr_escape_kinds": _tally(d["pnr_escape_kinds"] for d in found),
            "pnr_v5_reason": _tally(d["pnr_v5"]["v5_reason"] for d in found),
            "pnr_v2_outcome": _tally(d["pnr_v5"]["outcome"] for d in found),
            "pnr_taken_boost": sum(int(d["walk"]["taken_action"]) >= 3 for d in found),
            "boost_fatal_choice_same_direction_normal_status": _tally(
                d.get("pnr_same_direction_normal_status")
                for d in self_rows
                if d["category"] == "boost_fatal_choice"
            ),
            "boost_fatal_choice_landing_counts": [
                {
                    "taken": int(d["walk"]["taken_action"]),
                    "landing": d["pnr_v5"]["landing_counts"][int(d["walk"]["taken_action"]) % 3],
                    "need": d["pnr_v5"]["need"],
                }
                for d in self_rows
                if d["category"] == "boost_fatal_choice"
            ],
            "normal_fatal_choice_pnr_v2_outcome": _tally(
                d["pnr_v5"]["outcome"] for d in self_rows if d["category"] == "normal_fatal_choice"
            ),
            "escaping_actions_all_masked_legal": all(
                d.get("pnr_escaping_masked_legal") == d["walk"]["escaping_actions"] for d in found
            ),
            "fatal_lengths": sorted(int(d["fatal_length"]) for d in self_rows),
        }
    return summary


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def earlier_domains(domain: str) -> Dict[str, Sequence[str]]:
    """Every earlier domain/purpose to exclude (diagnose.py's set plus all v5 lanes)."""
    from research.apex_veto_v4_screen_20261001 import screen as v4
    from research.apex_veto_v5_screen_20261001 import screen as v5screen
    from research.apex_veto_v5_strict_20261001 import strict_run as v5strict

    earlier: Dict[str, Sequence[str]] = dict(v4.EARLIER_DOMAINS)
    earlier[v4.DOMAIN] = (v4.NAMESPACE,)
    earlier[v4.SMOKE_DOMAIN] = (v4.NAMESPACE,)
    earlier.update(v5screen.EARLIER_DOMAINS)
    earlier.update(v5strict.EARLIER_DOMAINS)
    earlier.update(V5_DOMAINS)
    for other in (DOMAIN, SMOKE_DOMAIN):
        if other != domain:
            earlier[other] = (NAMESPACE,)
    earlier.pop(domain, None)
    return earlier


def disjointness(seeds: Sequence[int], domain: str) -> Dict[str, Any]:
    from research.apex_safety_20260926 import dev_screen as ds

    extra = {
        f"{name}/{purpose}[0:{EXCLUSION_PREFIX}]": [
            ds.uint32_seed(name, purpose, i) for i in range(EXCLUSION_PREFIX)
        ]
        for name, purposes in earlier_domains(domain).items()
        for purpose in purposes
    }
    return ds.disjointness_report(
        seeds,
        ds.DEFAULT_PILOT_OUTPUT if ds.DEFAULT_PILOT_OUTPUT.is_dir() else None,
        screen_domain=domain,
        extra_namespaces=extra,
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", help="smoke namespace, any horizon")
    parser.add_argument("--worlds-per-mix", type=int, default=WORLDS_PER_MIX)
    parser.add_argument("--frames", type=int, default=None, help="smoke only")
    parser.add_argument("--mixes", default="frozen,scripted,mixed")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--budget", type=int, default=NODE_BUDGET)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        print(f"refusing: {out} already exists (create-only)", file=sys.stderr)
        return 2
    if not args.smoke and (args.frames is not None or args.worlds_per_mix != WORLDS_PER_MIX):
        print("refusing: --frames/--worlds-per-mix overrides are smoke-only", file=sys.stderr)
        return 2
    wrong_env = {k: os.environ.get(k) for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v}
    if wrong_env:
        print(f"refusing: environment must have {REQUIRED_ENV}, got {wrong_env}", file=sys.stderr)
        return 2
    if not base.on_ac_power():
        print("refusing: `pmset -g batt` does not show 'AC Power'", file=sys.stderr)
        return 2
    if not SLOT_LOCK.is_file():
        print(f"refusing: slot lock {SLOT_LOCK} missing (never created here)", file=sys.stderr)
        return 2
    lock = SLOT_LOCK.open("r")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print(f"refusing: CPU slot 1 busy ({SLOT_LOCK})", file=sys.stderr)
        return 2
    try:
        return _run(args, out, argv)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _run(args: argparse.Namespace, out: Path, argv: Sequence[str] | None) -> int:
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    from research.apex_safety_20260926 import dev_screen as ds
    from src.core.config_loader import load_and_initialize_config
    from src.evaluation.safety_veto_v5 import VETO_METHOD_V5
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.scripts.tournament_eval import evaluation_profile_for_name, rollout

    started_total = time.monotonic()
    domain = SMOKE_DOMAIN if args.smoke else DOMAIN
    horizon = int(args.frames) if (args.smoke and args.frames) else ds.HORIZON
    mixes = [m for m in args.mixes.split(",") if m]
    if any(m not in ds.MIXES for m in mixes):
        raise SystemExit(f"unknown mix in {mixes}")
    seeds = base.world_seeds(domain, int(args.worlds_per_mix))
    report = disjointness(seeds, domain)
    if not report["disjoint"]:
        raise SystemExit(f"world seeds overlap an earlier namespace: {report}")
    if ds.sha256_file(ds.DEFAULT_CONFIG) != ds.CONFIG_SHA256:
        raise SystemExit("deployment config bytes differ from dev_screen.CONFIG_SHA256")
    load_and_initialize_config(str(ds.DEFAULT_CONFIG))
    profile = evaluation_profile_for_name(ds.PROFILE_NAME, ds.HORIZON)
    if horizon != ds.HORIZON:
        from dataclasses import replace

        profile = replace(profile, scored_horizon=horizon)
    elif profile.digest != ds.PROFILE_DIGEST:
        raise SystemExit("resolved profile digest differs from dev_screen.PROFILE_DIGEST")
    sources = {
        name: ds.sha256_file(REPO / "src" / "evaluation" / name)
        for name in ("safety_veto.py", "safety_veto_v3.py", "safety_veto_v5.py")
    }
    rows_by_mix = {
        mix: [row for row in ds._design_rows(seeds) if row["mix"] == mix] for mix in mixes
    }
    out.mkdir(parents=True)
    for sub in ("deaths", "windows", "posts", "records"):
        (out / sub).mkdir()
    snapshots = ds.snapshot_checkpoints(ds.DEFAULT_CHECKPOINT_DIR, out)
    lookup = ds.agent_lookup(snapshots)
    hero = lookup[ds.CHAMPION[1]]
    base.write_new_json(
        out / "intent.json",
        {
            "schema_version": SCHEMA,
            "authority": AUTHORITY,
            "argv": list(sys.argv if argv is None else argv),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "git": base._git(),
            "smoke": bool(args.smoke),
            "worlds": {
                "domain": domain,
                "namespace": NAMESPACE,
                "recipe": "uint32 big-endian prefix of sha256('<domain>|<namespace>|<i>')",
                "seeds": seeds,
                "shared_by_mixes": mixes,
                "disjointness": report,
            },
            "config": {"path": str(ds.DEFAULT_CONFIG), "sha256": ds.CONFIG_SHA256},
            "profile": {"name": ds.PROFILE_NAME, "digest": profile.digest, "horizon": horizon},
            "checkpoint_snapshots": snapshots,
            "hero": {"name": ds.CHAMPION[0], "sha256": ds.CHAMPION[1]},
            "veto": {
                "method": VETO_METHOD_V5,
                "install": "dev_screen.hero_veto_installer(install_boost_aware_veto) "
                "(the v5 screen's arm B; rollout's vector61 guard runs first)",
                "source_sha256": sources,
            },
            "engine": {"engine": "live", "runner": "tournament_eval.rollout"},
            "capture": {
                "window_decisions": int(args.window),
                "decision_point": "installed v5 veto instance .apply (read-only wrapper)",
                "post_move_point": "GameLogic.check_collisions (read-only wrapper)",
            },
            "search": {
                "depth": int(args.depth),
                "node_budget_per_action": int(args.budget),
                "pnr": "diagnose.analyze_death (unchanged), self deaths only",
                "fatal_frame": "exact one-frame counterfactual vs the post-move world; "
                "escape from surviving alternatives with others static at post-move cells",
            },
            "threads": {
                "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
                "torch_intraop": torch.get_num_threads(),
                "torch_interop": torch.get_num_interop_threads(),
                "analysis_workers": int(args.workers),
            },
            "slot_lock": str(SLOT_LOCK),
            "ac_power_checked_at_start": True,
        },
    )

    jobs: List[Tuple[Tuple[str, str, Mapping[str, Any], str, int, int], Dict[str, Any]]] = []
    episodes: List[Dict[str, Any]] = []
    sim_batches: List[Dict[str, Any]] = []
    recorder = LiveRecorder(args.window)
    for mix in mixes:
        t_mix = time.monotonic()
        deaths_here = 0
        for index, row in enumerate(rows_by_mix[mix]):
            seed = int(row["world_seed"])
            opponents = [lookup[s["member_sha256"]] for s in row["slots"]]
            recorder.reset()
            t0 = time.monotonic()
            with live_capture(recorder):
                record = rollout(
                    hero,
                    opponents,
                    horizon,
                    seed,
                    profile=profile,
                    world_identity=_expected_world_identity(row),
                    mix_id=mix,
                    hero_safety_veto=True,
                )
            wall = time.monotonic() - t0
            probe = record["probes"]["safety_veto"]
            if probe["method"] != VETO_METHOD_V5:
                raise RuntimeError(f"hero veto is not v5: {probe['method']}")
            veto = recorder.veto
            cause = record["probes"].get("death_cause")
            entry = {
                "mix": mix,
                "world_index": index,
                "world_seed": seed,
                "deaths": record["deaths"],
                "death_cause": cause,
                "survival_fraction": record["survival_fraction"],
                "mass_integral": record["mass_integral"],
                "veto_decisions": probe["counters"]["decisions"],
                "captured_decisions": recorder.decisions,
                "v5_diagnostics": veto.diagnostics_record(),
                "wall_seconds": wall,
            }
            base.write_new_json(out / "records" / f"{mix}-{seed}.json", {**entry, "record": record})
            if entry["captured_decisions"] != entry["veto_decisions"]:
                raise RuntimeError(f"capture count differs from veto decisions: {entry}")
            episodes.append(entry)
            if float(record["deaths"]) >= 1:
                snaps = list(recorder.buffer)
                post = recorder.post
                if post is None or post["frame"] != snaps[-1]["frame"]:
                    raise RuntimeError(f"post-move capture is not the fatal frame: {entry}")
                window = out / "windows" / f"{mix}-{seed}.npz"
                np.savez_compressed(window, **pack_live_window(snaps))
                post_path = out / "posts" / f"{mix}-{seed}.json"
                base.write_new_json(post_path, post)
                meta = dict(recorder.world_meta or {})
                jobs.append(
                    (
                        (str(window), str(post_path), meta, str(cause), args.depth, args.budget),
                        entry,
                    )
                )
                deaths_here += 1
            print(
                json.dumps({"mix": mix, "seed": seed, "cause": cause, "wall_s": round(wall, 1)}),
                flush=True,
            )
        sim_batches.append(
            {
                "mix": mix,
                "episodes": len(rows_by_mix[mix]),
                "deaths": deaths_here,
                "wall_seconds": time.monotonic() - t_mix,
            }
        )
    sim_seconds = time.monotonic() - started_total

    t_an = time.monotonic()
    work = [job for job, _ in jobs]
    if int(args.workers) > 1 and len(work) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(int(args.workers)) as pool:
            results = pool.map(analyze_live_death_file, work, chunksize=1)
    else:
        results = [analyze_live_death_file(w) for w in work]
    analysis_seconds = time.monotonic() - t_an
    death_rows: List[Dict[str, Any]] = []
    for (job, entry), result in zip(jobs, results):
        row = {**entry, "cause": entry["death_cause"], **result}
        row["window_file"] = Path(job[0]).name
        base.write_new_json(out / "deaths" / f"{entry['mix']}-{entry['world_seed']}.json", row)
        death_rows.append(row)

    summary = summarize_census(episodes, death_rows)
    summary["timing"] = {
        "total_wall_seconds": time.monotonic() - started_total,
        "simulation_wall_seconds": sim_seconds,
        "analysis_wall_seconds": analysis_seconds,
        "simulation_batches": sim_batches,
    }
    summary.update(
        {
            "schema_version": SCHEMA,
            "authority": AUTHORITY,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "smoke": bool(args.smoke),
            "search": {"depth": int(args.depth), "node_budget_per_action": int(args.budget)},
        }
    )
    for sha, path in snapshots.items():
        if ds.sha256_file(Path(path)) != sha:
            raise RuntimeError(f"checkpoint snapshot {path} changed during the run")
    base.write_new_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("episode_outcomes_by_mix", "timing")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
