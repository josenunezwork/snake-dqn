"""B8 serving P2 and session-vs-grid parity probes.

The two probes in this module deliberately use a fresh T12 seed root.  They do
not reuse the development P2 namespace: those worlds already carry diagnostic
outcomes.  Both paths run one world at a time, CPU-thread bounded, and retain
the complete metric records while storing only hashes of the per-frame state.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT_NAME = "b8-s3"
CHECKPOINT_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
MIXES = ("frozen", "scripted", "mixed")
SIMD_ONLY = {
    "world_runtime_spec",
    "world_runtime_spec_digest",
    "vector61_policy",
    "veto_diagnostics",
    "ego2s_hero",
}


@contextlib.contextmanager
def _b8_environment() -> Iterator[None]:
    """Select B8 by the public serving selector and restore the caller's env."""
    keys = (
        "SNAKE_SERVE_CHECKPOINT",
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
        "SNAKE_SERVE_VETO_VARIANT",
    )
    saved = {key: os.environ.get(key) for key in keys}
    os.environ["SNAKE_SERVE_CHECKPOINT"] = CHECKPOINT_NAME
    # The protocol uses the ordinary serving defaults; no veto override is active.
    for key in keys[1:]:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _resolve_b8() -> Dict[str, Any]:
    """Resolve the selected registry name and prove the pinned B8 bytes are in use."""
    from web.backend.served_checkpoint import resolve_served_checkpoint

    with _b8_environment():
        choice = resolve_served_checkpoint()
    detail = choice.to_dict()
    if (
        detail["name"] != CHECKPOINT_NAME
        or detail["sha256"] != CHECKPOINT_SHA256
        or not detail["path"]
    ):
        raise RuntimeError(f"b8-s3 is not servable for qualification: {detail}")
    actual = hashlib.sha256(Path(str(detail["path"])).read_bytes()).hexdigest()
    if actual != CHECKPOINT_SHA256:
        raise RuntimeError(f"b8-s3 resolved bytes mismatch: {actual}")
    detail["verified_sha256"] = actual
    return detail


def _context(horizon: int) -> Dict[str, Any]:
    """Build the old P2 gate context with the requested, actual scored horizon."""
    import torch

    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    torch.set_num_threads(1)
    context = gid._context(1)
    if horizon != 5000:
        context["profile"] = dataclasses.replace(context["profile"], scored_horizon=int(horizon))
    return context


def _row_for_seed(seed: int, mix: str) -> Dict[str, Any]:
    """Materialize one gate roster for one independently audited world seed."""
    from research.apex_safety_20260926 import dev_screen

    if mix not in MIXES:
        raise ValueError(f"mix must be one of {MIXES}, got {mix!r}")
    rows = [row for row in dev_screen._design_rows([int(seed)]) if row["mix"] == mix]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one {mix} roster for world {seed}, got {len(rows)}")
    return rows[0]


def _assigned_rows(seeds: Sequence[int]) -> list[Dict[str, Any]]:
    """Assign audited actual world seeds to the registered 9/3 P2 mix layout."""
    if len(seeds) == 9:
        mixes = ("frozen",) * 3 + ("scripted",) * 3 + ("mixed",) * 3
    elif len(seeds) == 3:
        mixes = MIXES
    else:
        raise ValueError("P2 requires nine H5000 seeds or three H10000 seeds")
    if len({int(seed) for seed in seeds}) != len(seeds):
        raise ValueError("P2 actual world seeds must be distinct")
    return [_row_for_seed(int(seed), mix) for seed, mix in zip(seeds, mixes)]


def _roster(row: Mapping[str, Any], lookup: Mapping[str, Any]) -> list[Any]:
    return [lookup[slot["member_sha256"]] for slot in row["slots"]]


@contextlib.contextmanager
def _p2_horizon(horizon: int) -> Iterator[None]:
    """Use P2's live helper at the actual requested horizon, restoring global state."""
    from research.apex_safety_20260926 import dev_screen

    old = dev_screen.HORIZON
    dev_screen.HORIZON = int(horizon)
    try:
        yield
    finally:
        dev_screen.HORIZON = old


def _run_p2_live(
    row: Mapping[str, Any], context: Mapping[str, Any], student: str, horizon: int
) -> Dict[str, Any]:
    from research.redesign_m3_20261008 import student_record_parity as p2

    with _p2_horizon(horizon):
        return p2.run_live(row, context, student, "none")


def _run_p2_sim(
    row: Mapping[str, Any], context: Mapping[str, Any], student: str, horizon: int
) -> Dict[str, Any]:
    """The P2-NV grid call, intentionally one world per invocation."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env import eval_engine as ee

    seed = int(row["world_seed"])
    roster = _roster(row, context["lookup"])
    records = ee.run_simd_eval(
        context["lookup"][gid.FRP3_S12[1]],
        roster,
        int(horizon),
        [seed],
        profile=context["profile"],
        opponent_specs_by_world={seed: roster},
        mix_id=str(row["mix"]),
        vector61=True,
        vector61_forward="rowwise",
        sim_engine="grid",
        hero_ego2s=student,
        ego2s_forward="rowwise",
        ego2s_mask="advisory",
    )
    return records[0]


def run_p2(seeds: list[int], horizon: int) -> Dict[str, Any]:
    """Run no-veto B8 record parity for every supplied root seed and gate mix.

    The caller supplies the audited *actual* world seeds: nine for H5000
    (frozen x3, scripted x3, mixed x3), or three for H10000 (one per mix).
    Each live/grid pair is isolated to one world, avoiding cross-world forwards.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not seeds:
        raise ValueError("at least one fresh root seed is required")
    from research.redesign_m3_20261008 import student_record_parity as p2

    selected = _resolve_b8()
    context = _context(horizon)
    worlds: list[Dict[str, Any]] = []
    for row in _assigned_rows(seeds):
        student = str(selected["path"])
        live = _run_p2_live(row, context, student, horizon)
        sim = _run_p2_sim(row, context, student, horizon)
        comparison = p2.compare(live, sim)
        worlds.append(
            {
                "mix": row["mix"],
                "world_seed": int(row["world_seed"]),
                "horizon": int(horizon),
                "identical": bool(comparison["identical"]),
                "comparison": comparison,
                "live": live,
                "simd": sim,
                "world_identity": row.get("world_identity"),
            }
        )
    return {
        "schema_version": "t12-b8-serving-p2/v1",
        "checkpoint": selected,
        "actual_world_seeds": [int(seed) for seed in seeds],
        "horizon": int(horizon),
        "worlds": worlds,
        "compared": len(worlds),
        "identical": sum(bool(world["identical"]) for world in worlds),
        "pass": bool(worlds) and all(bool(world["identical"]) for world in worlds),
    }


def _session_signature(game: Any) -> list[Any]:
    frp3_run = import_module("research.frp3_checkpoint_serving_20261005.serving_run")

    return frp3_run.live_frame_signature(game)


def _simd_signature(sim: Any) -> list[Any]:
    frp3_run = import_module("research.frp3_checkpoint_serving_20261005.serving_run")

    return frp3_run.simd_frame_signature(sim, 0)


def _session_side(
    seed: int, horizon: int, row: Mapping[str, Any], selected: Mapping[str, Any]
) -> Dict[str, Any]:
    """Drive the real selected GameSession, replacing only its opponent roster."""
    from research.apex_veto_serving_20261001 import serving_run as v2run
    from src.scripts.eval_cli import set_seed
    from src.scripts.tournament_eval import _attach_agent
    from web.backend.session import GameSession

    context = _context(horizon)
    with v2run.parity_config(), _b8_environment():
        set_seed(int(seed))
        session = GameSession()
        choice = session.served_checkpoint or {}
        if choice.get("name") != CHECKPOINT_NAME or choice.get("sha256") != CHECKPOINT_SHA256:
            raise RuntimeError(f"GameSession did not serve b8-s3: {choice}")
        session.policy.forward = "rowwise"
        hero = session.game.snakes[0]
        hero.auto_respawn = False
        initial_veto = session.safety_veto_state()
        if initial_veto is None or initial_veto.get("active"):
            raise RuntimeError(f"b8-s3 must serve unwrapped: {initial_veto}")
        if "applies only to a vector61 Apex policy" not in str(initial_veto.get("reason")):
            raise RuntimeError(f"unexpected b8-s3 veto reason: {initial_veto}")
        policy_cache: Dict[str, Any] = {}
        for slot, spec in enumerate(_roster(row, context["lookup"]), start=1):
            _attach_agent(
                session.game, slot, spec, int(seed), policy_cache, profile=context["profile"]
            )
        # Rebuild the hero cache after the roster mutation so its exact live
        # prepare_frame path sees the same opponent types as the grid call.
        invalidate = getattr(session.policy, "_invalidate_cache", None)
        if callable(invalidate):
            invalidate()
        signatures = []
        started = time.perf_counter()
        for _ in range(int(horizon)):
            session.step()
            if session.last_error:
                raise RuntimeError(f"served session error: {session.last_error}")
            signatures.append(_digest(_session_signature(session.game)))
        return {
            "served_checkpoint": choice,
            "config_path": session.config_path,
            "config_sha256": hashlib.sha256(Path(session.config_path).read_bytes()).hexdigest(),
            "obs_spec": session.obs_spec,
            "veto": session.safety_veto_state(),
            "initial_veto": initial_veto,
            "last_error": session.last_error,
            "decisions": int(horizon),
            "frame_digests": signatures,
            "frame_chain_sha256": _digest(signatures),
            "terminal_hero": not bool(hero.is_alive),
            "hero_final_length": int(hero.length),
            "wall_seconds": time.perf_counter() - started,
        }


def _simd_side(
    seed: int, horizon: int, row: Mapping[str, Any], selected: Mapping[str, Any]
) -> Dict[str, Any]:
    """Run the matching advisory-mask, rowwise, no-veto GridBatchSim world."""
    from research.apex_veto_serving_20261001 import serving_run as v2run
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.core.config_loader import load_and_initialize_config
    from src.simd_env import eval_engine as ee

    context = _context(horizon)
    load_and_initialize_config(str(v2run.PARITY_CONFIG))
    roster = _roster(row, context["lookup"])
    signatures: list[str] = []
    original = ee._hero_post_frame_snapshot

    def traced(sim: Any, events: Any) -> Any:
        signatures.append(_digest(_simd_signature(sim)))
        return original(sim, events)

    ee._hero_post_frame_snapshot = traced
    started = time.perf_counter()
    try:
        record = ee.run_simd_eval(
            context["lookup"][gid.FRP3_S12[1]],
            roster,
            int(horizon),
            [int(seed)],
            profile=context["profile"],
            opponent_specs_by_world={int(seed): roster},
            mix_id=str(row["mix"]),
            frame_observer=lambda _frame: None,
            vector61=True,
            vector61_forward="rowwise",
            sim_engine="grid",
            hero_ego2s=str(selected["path"]),
            ego2s_forward="rowwise",
            ego2s_mask="advisory",
        )[0]
    finally:
        ee._hero_post_frame_snapshot = original
    return {
        "record": record,
        "frame_digests": signatures,
        "frame_chain_sha256": _digest(signatures),
        "wall_seconds": time.perf_counter() - started,
    }


def run_session_parity(seed: int, horizon: int = 5000, mix: str = "frozen") -> Dict[str, Any]:
    """Compare the served B8 session against the corresponding one-world grid evaluation."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if mix not in MIXES:
        raise ValueError(f"mix must be one of {MIXES}, got {mix!r}")
    selected = _resolve_b8()
    row = _row_for_seed(int(seed), mix)
    live = _session_side(int(seed), int(horizon), row, selected)
    simd = _simd_side(int(seed), int(horizon), row, selected)
    left, right = live["frame_digests"], simd["frame_digests"]
    compared = min(len(left), len(right))
    first = next((index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)
    if first is None and len(left) != len(right):
        first = compared
    result = {
        "schema_version": "t12-b8-serving-session-parity/v1",
        "checkpoint": selected,
        "world_seed": int(seed),
        "mix": mix,
        "horizon": int(horizon),
        "world_identity": row.get("world_identity"),
        "session": live,
        "simd": simd,
        "frames_compared": compared,
        "session_frame_count": len(left),
        "simd_frame_count": len(right),
        "first_differing_frame": first,
        "session_trace_sha256": _digest(left),
        "simd_trace_sha256": _digest(right),
    }
    result["pass"] = first is None and result["session_trace_sha256"] == result["simd_trace_sha256"]
    return result
