"""Live serving measurements for the B8-s3 qualification.

This module deliberately drives :class:`web.backend.session.GameSession`, rather
than a simulator or a model directly.  It is therefore suitable for the serving
qualification only after its pinned protocol is in force.  The latency entry
point refuses to start while the T11 training tmux session exists.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import random
import re
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence
from unittest.mock import patch

import numpy as np
import torch

from src.core.game_config import get_config, initialize_config
from web.backend.session import MODE_PLAY, MODE_WATCH, GameSession
from research.t12_b8_serving_20261017.episodes import ScriptedHuman, _control, set_world_seed

CHECKPOINT_NAME = "b8-s3"
UI_SPEED_MAX = 60


class LatencyBlocked(RuntimeError):
    """Raised when the required quiet-machine assertion cannot be established."""


def frontend_decision_margin(q_values: Sequence[object]) -> Dict[str, float]:
    """Mirror ``frontend/src/lib/decisionMargin.ts`` without a browser runtime."""
    finite = sorted((float(value) for value in q_values if _is_finite_number(value)), reverse=True)
    if len(finite) < 2:
        return {"raw": 0.0, "relative": 0.0}
    raw = finite[0] - finite[1]
    spread = finite[0] - finite[-1]
    return {"raw": raw, "relative": raw / spread if spread > 1e-9 else 0.0}


def confidence_bucket(relative: float) -> str:
    """Return the frontend confidence bucket for a relative decision margin."""
    if relative > 0.25:
        return "confident"
    if relative > 0.1:
        return "moderate"
    return "close"


def run_confidence(seed: int, frames: int = 1500) -> Dict[str, object]:
    """Collect the frontend confidence split from served B8 Watch snapshots."""
    _require_positive_frames(frames)
    with _saved_rng_state(), _served_b8_environment(), _saved_global_config():
        _seed_everything(seed)
        set_world_seed(seed)
        session = GameSession()
        _assert_b8_session(session)
        buckets = {"confident": 0, "moderate": 0, "close": 0}
        observed = 0
        inspector_groups: List[Dict[str, object]] = []
        veto_states: List[object] = []
        attempts = 0
        missing_inspector = 0
        dead_hero = 0
        nonfinite_values = 0
        q_vectors: List[List[float]] = []
        while observed < frames and attempts < 3000:
            attempts += 1
            session.step()
            snapshot = session.snapshot()
            inspector = snapshot.get("inspector") or {}
            q_values = inspector.get("q_values")
            if isinstance(q_values, list):
                nonfinite_values += sum(1 for value in q_values if not _is_finite_number(value))
                q_vectors.append([float(value) for value in q_values])
                margin = frontend_decision_margin(q_values)
                buckets[confidence_bucket(margin["relative"])] += 1
                observed += 1
            else:
                missing_inspector += 1
                if not inspector:
                    dead_hero += 1
            if not inspector_groups and isinstance(inspector.get("groups"), list):
                inspector_groups = [dict(group) for group in inspector["groups"]]
            veto_states.append(session.safety_veto_state())
        if observed != frames:
            raise RuntimeError(
                f"confidence collection incomplete: q_values on {observed}/{frames} Watch frames"
            )
        veto_evidence = _no_veto_evidence(veto_states)
        return {
            "checkpoint": CHECKPOINT_NAME,
            "mode": MODE_WATCH,
            "seed": int(seed),
            "frames_requested": int(frames),
            "frames_with_q_values": observed,
            "attempted_frames": attempts,
            "missing_inspector_frames": missing_inspector,
            "dead_hero_inspector_frames": dead_hero,
            "nonfinite_q_values": nonfinite_values,
            "buckets": buckets,
            "q_values": q_vectors,
            "inspector_groups": inspector_groups,
            "no_veto": veto_evidence,
            "served": _served_identity(session),
        }


def run_latency(seeds: List[int], frames: int = 1500, warmup: int = 200) -> Dict[str, object]:
    """Measure B8 game-update and full serving-tick costs in four live cases.

    The full tick is exactly ``GameSession.step(); GameSession.snapshot()``, the
    same order as ``Hub.engine_loop``.  ``update_ms`` is measured by wrapping the
    actual ``GameState.update`` call inside that tick.  No pacing happens during
    collection.  The separate paced probe reproduces Hub's sleep-after-work
    behavior and reports overruns rather than claiming a requested FPS was met.
    """
    _require_positive_frames(frames)
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if len(seeds) != 4:
        raise ValueError("latency requires exactly four registered seeds, one per case")
    preconditions = latency_preconditions()
    cases = ((MODE_WATCH, 6), (MODE_WATCH, 12), (MODE_PLAY, 6), (MODE_PLAY, 12))
    with _served_b8_environment(), _saved_global_config(), _saved_rng_state():
        results = []
        for (mode, snakes), seed in zip(cases, seeds):
            result = _measure_latency_case(mode, snakes, int(seed), frames, warmup)
            final_bound, pacing_attempts = _validate_pacing(
                result["ui_speed_bound_fps"], int(seed), mode, snakes, warmup
            )
            result["ui_speed_bound_fps"] = final_bound
            result["pacing_attempts"] = pacing_attempts
            result["paced_probe"] = pacing_attempts[-1]
            results.append(result)
    chosen = min(item["ui_speed_bound_fps"] for item in results)
    return {
        "checkpoint": CHECKPOINT_NAME,
        "frames_per_seed": int(frames),
        "warmup_per_session": int(warmup),
        "preconditions": preconditions,
        "cases": results,
        "shared_ui_speed_fps": chosen,
        "definitions": {
            "update_ms": "wall time inside GameState.update",
            "tick_ms": "wall time for GameSession.step followed by snapshot",
            "ui_speed_bound": "largest slider setting 1..60 where max tick < 1000/fps",
            "p95_capacity_fps": (
                "floor(1000/p95 tick ms), reported separately from no-overrun bound"
            ),
            "estimated_achieved_fps": (
                "1/(1/requested_fps + mean_tick_seconds), Hub sleep-after-work model"
            ),
        },
    }


def latency_preconditions() -> Dict[str, object]:
    """Assert immediate T11 absence and record the Mac power/lid/process state."""
    tmux = _tmux_t11_absent()
    power = _mac_power_and_lid()
    processes = _process_observation()
    if not power["ac_power"]:
        raise LatencyBlocked("latency blocked: Mac is not on AC power")
    if power["lid_closed"] is not False:
        raise LatencyBlocked("latency blocked: Mac lid state is not confirmed open")
    return {"tmux": tmux, "machine": power, "processes": processes}


def _measure_latency_case(
    mode: str, snakes: int, seed: int, frames: int, warmup: int
) -> Dict[str, object]:
    all_update: List[float] = []
    all_tick: List[float] = []
    restarts = 0
    _seed_everything(seed)
    set_world_seed(seed)
    session = _new_session(mode, snakes)
    _assert_b8_session(session)
    stand_in = ScriptedHuman(seed) if mode == MODE_PLAY else None
    collected = 0
    session_warmup = warmup
    attempts = 0
    warmup_advancing_frames = 0
    max_attempts = max(10_000, (frames + warmup + 1) * 20)
    while collected < frames:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Play latency could not obtain warmed live frames before attempt limit"
            )
        if stand_in is not None:
            _stand_in_human_control(session, stand_in)
        before_frame = int(session.game.frame)
        update_ms, tick_ms, snapshot = _timed_tick(session)
        _assert_live_advance(session, before_frame)
        if session_warmup:
            session_warmup -= 1
            warmup_advancing_frames += 1
        else:
            all_update.append(update_ms)
            all_tick.append(tick_ms)
            collected += 1
            if not isinstance(snapshot, Mapping):
                raise RuntimeError("GameSession.snapshot did not return a mapping")
        # A terminal human frame advanced the live game and is therefore a
        # valid warmup/measurement sample.  Reset only after accounting for it.
        if mode == MODE_PLAY and session.run_over:
            restarts += 1
            session.reset_game()
            session_warmup = warmup
    tick_stats = _distribution(all_tick)
    max_tick = tick_stats["max"]
    p95_tick = tick_stats["p95"]
    ui_bound = _highest_ui_speed(max_tick)
    return {
        "case": f"{mode.title()}{snakes}",
        "mode": mode,
        "snakes": snakes,
        "seed": int(seed),
        "attempts": attempts,
        "restart_count": restarts,
        "advancing_measured_frames": len(all_tick),
        "advancing_warmup_frames": warmup_advancing_frames,
        "served": _served_identity(session),
        "roster": _roster_identity(session),
        "last_error": session.last_error,
        "samples": {"update_ms": all_update, "tick_ms": all_tick},
        "update_ms": _distribution(all_update),
        "tick_ms": tick_stats,
        "ui_speed_bound_fps": ui_bound,
        "p95_capacity_fps": int(math.floor(1000.0 / p95_tick)) if p95_tick > 0 else None,
        "estimated_achieved_fps": _estimated_achieved_fps(ui_bound, tick_stats["mean"]),
    }


def _paced_probe(fps: int, seed: int, mode: str, snakes: int, warmup: int) -> Dict[str, object]:
    """Run 120 Hub-order ticks with sleep-after-work pacing for one roster."""
    _seed_everything(seed)
    set_world_seed(seed)
    session = _new_session(mode, snakes)
    _assert_b8_session(session)
    stand_in = ScriptedHuman(seed) if mode == MODE_PLAY else None
    warmup_evidence = _advance_warmup(session, stand_in, warmup)
    budget_s = 1.0 / fps
    tick_ms: List[float] = []
    interval_ms: List[float] = []
    overruns = 0
    restarts = 0
    started = time.perf_counter()
    previous = started
    for _ in range(120):
        tick_started = time.perf_counter()
        interval_ms.append((tick_started - previous) * 1000.0)
        if stand_in is not None:
            _stand_in_human_control(session, stand_in)
        before_frame = int(session.game.frame)
        _, elapsed_ms, _ = _timed_tick(session)
        _assert_live_advance(session, before_frame)
        if mode == MODE_PLAY and session.run_over:
            # This sample is valid; warm a new run before its next sample.
            restarts += 1
            session.reset_game()
            new_warmup = _advance_warmup(session, stand_in, warmup)
            warmup_evidence["advancing_frames"] += new_warmup["advancing_frames"]
            warmup_evidence["restarts"] += new_warmup["restarts"]
        tick_ms.append(elapsed_ms)
        if elapsed_ms / 1000.0 >= budget_s:
            overruns += 1
        # Hub sleeps the complete budget *after* it has done its work.
        time.sleep(budget_s)
        previous = tick_started
    elapsed_s = time.perf_counter() - started
    return {
        "requested_fps": fps,
        "mode": mode,
        "snakes": snakes,
        "ticks": 120,
        "raw_tick_ms": tick_ms,
        "tick_samples_ms": tick_ms,
        "interval_samples_ms": interval_ms[1:],
        "tick_ms": _distribution(tick_ms),
        "interval_ms": _distribution(interval_ms[1:]),
        "elapsed_seconds": elapsed_s,
        "overruns": overruns,
        "restart_count": restarts + warmup_evidence["restarts"],
        "advancing_warmup_frames": warmup_evidence["advancing_frames"],
        "served": _served_identity(session),
        "roster": _roster_identity(session),
        "last_error": session.last_error,
        "actual_achieved_fps": 120.0 / elapsed_s,
        "estimated_achieved_fps": _estimated_achieved_fps(fps, statistics.fmean(tick_ms)),
    }


def _validate_pacing(
    initial_bound: int, seed: int, mode: str, snakes: int, warmup: int
) -> tuple[int, List[Dict[str, object]]]:
    """Keep the initial and, at most, one lowered paced validation attempt."""
    attempts = [_paced_probe(initial_bound, seed, mode, snakes, warmup)]
    if attempts[0]["overruns"]:
        retry_bound = _highest_ui_speed(attempts[0]["tick_ms"]["max"])
        attempts.append(_paced_probe(retry_bound, seed, mode, snakes, warmup))
    for attempt in attempts:
        attempt["validated"] = attempt["overruns"] == 0
    return int(attempts[-1]["requested_fps"]), attempts


def _new_session(mode: str, snakes: int) -> GameSession:
    if mode == MODE_WATCH and snakes == 12:
        original_loader = __import__("web.backend.session", fromlist=["load_config"]).load_config

        def twelve_snake_config(path: str):
            config = original_loader(path)
            return replace(config, game=replace(config.game, num_snakes=12))

        with patch("web.backend.session.load_config", side_effect=twelve_snake_config):
            return GameSession()
    session = GameSession()
    if mode == MODE_PLAY:
        session.set_play_opponents(snakes - 1)
        session.set_mode(MODE_PLAY)
    return session


def _timed_tick(session: GameSession) -> tuple[float, float, dict]:
    original_update = session.game.update
    update_ms = 0.0

    def timed_update(*args, **kwargs):
        nonlocal update_ms
        started = time.perf_counter()
        try:
            return original_update(*args, **kwargs)
        finally:
            update_ms = (time.perf_counter() - started) * 1000.0

    session.game.update = timed_update
    started = time.perf_counter()
    try:
        session.step()
        snapshot = session.snapshot()
    finally:
        session.game.update = original_update
    return update_ms, (time.perf_counter() - started) * 1000.0, snapshot


def _stand_in_human_control(session: GameSession, stand_in: ScriptedHuman) -> None:
    """Drive Play through the same public control dispatch a browser uses."""
    human = session._find_human()
    if human is None or not human.is_alive:
        return
    direction = stand_in.choose(human, session.game, must_turn=not session.run_started)
    if direction is not None:
        _control(session, "human_input", direction)


def _advance_warmup(
    session: GameSession, stand_in: ScriptedHuman | None, warmup: int
) -> Dict[str, int]:
    """Advance exactly ``warmup`` live frames, restarting scored Play runs as needed."""
    advanced = 0
    attempts = 0
    advancing_frames = 0
    restarts = 0
    max_attempts = max(10_000, (warmup + 1) * 20)
    while advanced < warmup:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "paced latency warmup could not obtain live frames before attempt limit"
            )
        if stand_in is not None:
            _stand_in_human_control(session, stand_in)
        before_frame = int(session.game.frame)
        session.step()
        session.snapshot()
        _assert_live_advance(session, before_frame)
        advancing_frames += 1
        if session.mode == MODE_PLAY and session.run_over:
            session.reset_game()
            restarts += 1
            # Each fresh scored run receives a full 200-frame warmup.
            advanced = 0
            continue
        advanced += 1
    return {"advancing_frames": advancing_frames, "restarts": restarts}


def _distribution(samples: Sequence[float]) -> Dict[str, float]:
    if not samples:
        raise RuntimeError("measurement collected no samples")
    ordered = sorted(float(sample) for sample in samples)
    return {
        "count": len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    index = (len(ordered) - 1) * quantile
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (index - low))


def _highest_ui_speed(max_tick_ms: float) -> int:
    for fps in range(UI_SPEED_MAX, 0, -1):
        if max_tick_ms < 1000.0 / fps:
            return fps
    return 1


def _estimated_achieved_fps(requested_fps: int, mean_tick_ms: float) -> float:
    return 1.0 / (1.0 / requested_fps + mean_tick_ms / 1000.0)


def _no_veto_evidence(states: Sequence[object]) -> Dict[str, object]:
    active = [state for state in states if isinstance(state, Mapping) and state.get("active")]
    counters = [state.get("counters") for state in states if isinstance(state, Mapping)]
    return {"active_frames": len(active), "counter_snapshots": counters, "absent": not active}


def _assert_b8_session(session: GameSession) -> None:
    choice = getattr(session, "served_checkpoint", None) or {}
    if choice.get("name") != CHECKPOINT_NAME:
        raise RuntimeError(f"expected registry-selected {CHECKPOINT_NAME}, got {choice!r}")
    if str(getattr(session, "obs_spec", "")) != "ego2s-bv8":
        raise RuntimeError(f"B8 must serve ego2s-bv8, got {session.obs_spec!r}")
    if type(getattr(session, "policy", None)).__name__ != "Ego2sServingPolicy":
        raise RuntimeError(f"B8 must use Ego2sServingPolicy, got {type(session.policy).__name__}")
    veto = session.safety_veto_state() or {}
    if veto.get("active") or veto.get("wrapped_snake_ids") or veto.get("counters"):
        raise RuntimeError(f"B8 serving must have no veto wrapper/counters: {veto!r}")
    human_id = getattr(session, "human_id", None)
    ai = [snake for snake in session.game.snakes if int(snake.id) != human_id]
    if len(ai) == 0 or any(getattr(snake, "policy", None) is not session.policy for snake in ai):
        raise RuntimeError("every AI must share the registry-selected B8 serving policy")
    if any(getattr(snake, "safety_veto", None) is not None for snake in ai):
        raise RuntimeError("B8 AI roster must not carry a veto wrapper")


def _served_identity(session: GameSession) -> Dict[str, object]:
    """Small self-contained identity record kept beside raw Q/timing evidence."""
    path = getattr(session, "checkpoint_path", None)
    digest = None
    if path and Path(path).is_file():
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        "registry": dict(getattr(session, "served_checkpoint", None) or {}),
        "checkpoint_path": path,
        "checkpoint_sha256": digest,
        "obs_spec": str(getattr(session, "obs_spec", "")),
        "policy_class": type(getattr(session, "policy", None)).__name__,
        "mode": str(getattr(session, "mode", "")),
        "veto": session.safety_veto_state(),
    }


def _roster_identity(session: GameSession) -> Dict[str, object]:
    """Record the actual roster properties the latency protocol requires."""
    human_id = getattr(session, "human_id", None)
    ai = [snake for snake in session.game.snakes if int(snake.id) != human_id]
    return {
        "total_snakes": len(session.game.snakes),
        "human_id": human_id,
        "ai_ids": [int(snake.id) for snake in ai],
        "ai_count": len(ai),
        "all_ai_share_served_policy": all(
            getattr(snake, "policy", None) is session.policy for snake in ai
        ),
        "ai_veto_wrappers": [
            int(snake.id) for snake in ai if getattr(snake, "safety_veto", None) is not None
        ],
    }


def _assert_live_advance(session: GameSession, before_frame: int) -> None:
    """Reject paused/frozen tails and surface a serving error at its first frame."""
    if session.last_error:
        raise RuntimeError(f"serving error during measurement: {session.last_error}")
    if int(session.game.frame) <= before_frame:
        raise RuntimeError(
            f"measurement frame did not advance: before={before_frame} after={session.game.frame}"
        )


@contextlib.contextmanager
def _served_b8_environment() -> Iterator[None]:
    keys = (
        "SNAKE_SERVE_CHECKPOINT",
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
        "SNAKE_SERVE_VETO_VARIANT",
    )
    old = {key: os.environ.pop(key) for key in keys if key in os.environ}
    os.environ["SNAKE_SERVE_CHECKPOINT"] = CHECKPOINT_NAME
    try:
        yield
    finally:
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update(old)


@contextlib.contextmanager
def _saved_global_config() -> Iterator[None]:
    before = get_config()
    try:
        yield
    finally:
        initialize_config(before)


@contextlib.contextmanager
def _saved_rng_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _require_positive_frames(frames: int) -> None:
    if int(frames) <= 0:
        raise ValueError("frames must be positive")


def _tmux_t11_absent() -> Dict[str, object]:
    try:
        has_session = subprocess.run(
            ["tmux", "has-session", "-t", "=t11-lanes"], capture_output=True, text=True, check=False
        )
        if has_session.returncode == 0:
            panes = subprocess.run(
                [
                    "tmux",
                    "list-panes",
                    "-s",
                    "-t",
                    "=t11-lanes",
                    "-F",
                    "#{pane_dead}\t#{pane_pid}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if panes.returncode != 0:
                raise LatencyBlocked("latency blocked: cannot inspect present t11-lanes panes")
            pane_rows = [line.split("\t", 1) for line in panes.stdout.splitlines() if line.strip()]
            if not pane_rows or not all(row[0] == "1" for row in pane_rows):
                raise LatencyBlocked(
                    "latency blocked: tmux session t11-lanes still has live workload panes"
                )
            return {
                "session": "t11-lanes",
                "present": True,
                "workload_panes_exited": True,
                "has_session_returncode": has_session.returncode,
                "panes": [
                    {"dead": row[0], "pid": row[1] if len(row) > 1 else None} for row in pane_rows
                ],
            }
        listed = subprocess.run(
            ["tmux", "list-sessions"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise LatencyBlocked(f"latency blocked: cannot assert t11-lanes absence: {exc}") from exc
    names = [line.split(":", 1)[0] for line in listed.stdout.splitlines() if ":" in line]
    absent_diagnostic = "can't find session" in (has_session.stderr or "").lower()
    no_server_diagnostic = "no server running" in (listed.stderr or "").lower()
    if "t11-lanes" in names:
        raise LatencyBlocked("latency blocked: tmux list-sessions still contains t11-lanes")
    if not (absent_diagnostic or no_server_diagnostic):
        raise LatencyBlocked(
            "latency blocked: tmux did not provide an absence diagnostic for t11-lanes"
        )
    return {
        "session": "t11-lanes",
        "present": False,
        "has_session_returncode": has_session.returncode,
        "list_sessions_returncode": listed.returncode,
        "sessions": names,
        "has_session_stderr": has_session.stderr.strip(),
        "list_sessions_stderr": listed.stderr.strip(),
    }


def _mac_power_and_lid() -> Dict[str, object]:
    try:
        power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False)
        lid = subprocess.run(
            ["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise LatencyBlocked(f"latency blocked: cannot inspect Mac power/lid state: {exc}") from exc
    match = re.search(r'"AppleClamshellState"\s*=\s*(Yes|No)', lid.stdout, flags=re.IGNORECASE)
    lid_closed = True if match and match.group(1).lower() == "yes" else False if match else None
    return {
        "ac_power": "ac power" in power.stdout.lower(),
        "lid_closed": lid_closed,
        "pmset": power.stdout.strip(),
        "ioreg": lid.stdout.strip(),
    }


def _process_observation() -> Dict[str, object]:
    try:
        result = subprocess.run(
            ["ps", "-Ao", "pid,ppid,pcpu,command"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        return {"error": str(exc)}
    lines = [
        line
        for line in result.stdout.splitlines()
        if "snake" in line.lower() or "python" in line.lower()
    ]
    return {"returncode": result.returncode, "matching_processes": lines}
