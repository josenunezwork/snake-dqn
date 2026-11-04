"""Mac serving qualification on the exact Play path for Classic B8 (all_b8) and Mixed rosters.

Protocol:
- 200-frame warmup + 1,500 measured ticks at N in {1, 3, 5, 8, 11} for both
  all_b8 and mixed Play rosters on the exact GameSession Play path.
- Reports update_ms (p50/p95/p99/max/mean), full_tick_ms including human_input
  control handling + GameSession.step() + GameSession.snapshot() (p50/p95/p99/max/mean),
  step_snapshot_ms, and overruns against the current speed ceilings
  (24 fps for <=6 AI opponents, 12 fps for >6 AI opponents), plus a 120-tick
  paced probe at the clamped ceiling.
- Runs a 10-switch Play <-> Watch RSS check to verify no memory leak across mode rebuilds.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence
from unittest.mock import patch

for _env_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ[_env_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from research.apex_veto_serving_20261001.serving_run import (  # noqa: E402
    ScriptedHuman,
    set_world_seed,
)
from src.core.game_config import get_config, initialize_config  # noqa: E402
from web.backend import app as app_module  # noqa: E402
from web.backend import session as session_module  # noqa: E402
from web.backend.session import MODE_PLAY, MODE_WATCH, GameSession  # noqa: E402

WARMUP_FRAMES = 200
MEASURED_TICKS = 1500
PACED_TICKS = 120
OPPONENT_COUNTS = (1, 3, 5, 8, 11)
ROSTERS = ("all_b8", "mixed")
CONN_ID = "play-mixed-qual"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    index = (len(ordered) - 1) * quantile
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (index - low))


def _distribution(samples: Sequence[float]) -> Dict[str, float]:
    if not samples:
        raise RuntimeError("no samples collected")
    ordered = sorted(float(x) for x in samples)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _current_rss_bytes() -> int:
    """Return current resident set size (bytes) for this process via ps."""
    out = subprocess.check_output(
        ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())], text=True
    ).strip()
    return int(out) * 1024


def _max_rss_bytes() -> int:
    """Return peak resident set size (bytes on macOS) via getrusage."""
    val = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return val if sys.platform == "darwin" else val * 1024


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


@contextlib.contextmanager
def _isolated_serving_env(temp_dir: str) -> Iterator[None]:
    keys = (
        "SNAKE_SERVE_CHECKPOINT",
        "SNAKE_SERVE_WATCH_ROSTER",
        "SNAKE_SERVE_PLAY_ROSTER",
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
        "SNAKE_SERVE_VETO_VARIANT",
        "SNAKE_WATCH_SOURCE",
        "SNAKE_SCORES_DB",
    )
    saved = {k: os.environ.get(k) for k in keys}
    prev_cfg = get_config()
    try:
        for k in keys:
            os.environ.pop(k, None)
        os.environ["SNAKE_SERVE_CHECKPOINT"] = "b8-s3"
        os.environ["SNAKE_SERVE_WATCH_ROSTER"] = "mixed"
        os.environ["SNAKE_WATCH_SOURCE"] = "live"
        os.environ["SNAKE_SCORES_DB"] = str(Path(temp_dir) / "qual_scores.db")
        with (
            patch.object(session_module, "served_resolution"),
            patch.object(session_module, "watch_tick"),
            patch.object(session_module, "play_tick"),
            patch.object(session_module, "ego2s_exception"),
            patch.object(session_module, "mixed_roster_fallback"),
        ):
            yield
    finally:
        initialize_config(prev_cfg)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _control(session: GameSession, action: str, value: Any = None) -> None:
    session.last_error = None
    reply = app_module._apply_control(
        session, {"type": "control", "action": action, "value": value}, CONN_ID
    )
    if reply is not None or session.last_error is not None:
        raise RuntimeError(
            f"control {action!r} ({value!r}) failed: "
            f"reply={reply!r} last_error={session.last_error!r}"
        )


def _verify_play_session(session: GameSession, roster: str, opponents: int) -> Dict[str, Any]:
    if session.mode != MODE_PLAY:
        raise RuntimeError(f"expected mode='play', got {session.mode!r}")
    if (session.served_checkpoint or {}).get("name") != "b8-s3":
        raise RuntimeError(f"expected served b8-s3, got {session.served_checkpoint!r}")
    if session.play_roster != roster:
        raise RuntimeError(f"expected play_roster={roster!r}, got {session.play_roster!r}")
    if session.play_roster_reason is not None:
        raise RuntimeError(f"unexpected play_roster_reason: {session.play_roster_reason!r}")
    if len(session.game.snakes) != opponents + 1:
        raise RuntimeError(f"expected {opponents + 1} snakes, got {len(session.game.snakes)}")
    expected_ceiling = 24.0 if opponents <= 6 else 12.0
    if float(session.speed_ceiling) != expected_ceiling:
        raise RuntimeError(
            f"expected speed_ceiling={expected_ceiling}, got {session.speed_ceiling}"
        )
    human = session._find_human()
    if human is None or int(human.id) != 0:
        raise RuntimeError("missing human snake in slot 0")
    ai_snakes = list(session.game.snakes[1:])
    if roster == "all_b8":
        if any(s.policy is not session.policy for s in ai_snakes):
            raise RuntimeError("all_b8 Play AI snakes must all share session.policy")
        if any(s.safety_veto is not None for s in ai_snakes):
            raise RuntimeError("all_b8 Play AI snakes must be unwrapped")
    else:
        if ai_snakes[0].policy is not session.policy or ai_snakes[0].safety_veto is not None:
            raise RuntimeError("mixed Play slot 1 must be unwrapped bare b8-s3")
        for s in ai_snakes[1:]:
            if s.policy is not session.play_opponent_policy or s.safety_veto is None:
                raise RuntimeError("mixed Play slots 2..N must be wrapped FRP3+v8")
    veto = session.safety_veto_state() or {}
    return {
        "play_roster": session.play_roster,
        "opponents": opponents,
        "total_snakes": len(session.game.snakes),
        "speed_ceiling_fps": float(session.speed_ceiling),
        "b8_ai_count": sum(1 for s in ai_snakes if s.policy is session.policy),
        "frp3_v8_ai_count": sum(1 for s in ai_snakes if s.policy is session.play_opponent_policy),
        "wrapped_snake_ids": list(veto.get("wrapped_snake_ids") or []),
    }


def _full_play_tick(
    session: GameSession, stand_in: ScriptedHuman
) -> tuple[float, float, float, float]:
    """Execute one full Play tick: human_input control + step() + snapshot().

    Returns:
        (update_ms, step_snapshot_ms, full_tick_ms, input_ms)
    """
    human = session._find_human()
    if human is None or not human.is_alive:
        raise RuntimeError("human is not alive at start of Play tick")

    tick_start = time.perf_counter()
    direction = stand_in.choose(human, session.game, must_turn=not session.run_started)
    if direction is not None:
        _control(session, "human_input", direction)
    if not session.run_started:
        # Ensure the run is started on the first tick if choose() returned None
        for fallback_dir in ("up", "down", "right", "left"):
            _control(session, "human_input", fallback_dir)
            if session.run_started:
                break
    input_ms = (time.perf_counter() - tick_start) * 1000.0

    original_update = session.game.update
    update_ms = 0.0

    def timed_update(*args: Any, **kwargs: Any) -> Any:
        nonlocal update_ms
        u_start = time.perf_counter()
        try:
            return original_update(*args, **kwargs)
        finally:
            update_ms = (time.perf_counter() - u_start) * 1000.0

    before_frame = int(session.game.frame)
    step_start = time.perf_counter()
    session.game.update = timed_update
    try:
        session.step()
        frame = session.snapshot()
    finally:
        session.game.update = original_update
    step_snapshot_ms = (time.perf_counter() - step_start) * 1000.0
    full_tick_ms = (time.perf_counter() - tick_start) * 1000.0

    if session.last_error:
        raise RuntimeError(f"session error during Play tick: {session.last_error}")
    if int(session.game.frame) != before_frame + 1:
        raise RuntimeError(
            f"Play frame did not advance: before={before_frame} after={session.game.frame}"
        )
    if (
        not isinstance(frame, dict)
        or frame.get("play", {}).get("play_roster") != session.play_roster
    ):
        raise RuntimeError("snapshot missing play_roster state")

    return update_ms, step_snapshot_ms, full_tick_ms, input_ms


def measure_play_case(
    roster: str, opponents: int, seed: int, warmup: int = WARMUP_FRAMES, ticks: int = MEASURED_TICKS
) -> Dict[str, Any]:
    _seed_everything(seed)
    set_world_seed(seed)
    session = GameSession()
    _control(session, "set_mode", "play")
    _control(session, "set_play_opponents", opponents)
    _control(session, "set_play_roster", roster)
    _control(session, "new_game")
    roster_info = _verify_play_session(session, roster, opponents)

    stand_in = ScriptedHuman(seed)
    update_samples: List[float] = []
    step_snapshot_samples: List[float] = []
    full_tick_samples: List[float] = []
    input_samples: List[float] = []

    warmup_remaining = warmup
    warmup_frames_total = 0
    restarts = 0
    attempts = 0
    max_attempts = (ticks + warmup + 1) * 25

    while len(full_tick_samples) < ticks:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"exceeded max_attempts={max_attempts} for roster={roster} N={opponents}"
            )
        u_ms, ss_ms, ft_ms, in_ms = _full_play_tick(session, stand_in)
        if warmup_remaining > 0:
            warmup_remaining -= 1
            warmup_frames_total += 1
        else:
            update_samples.append(u_ms)
            step_snapshot_samples.append(ss_ms)
            full_tick_samples.append(ft_ms)
            input_samples.append(in_ms)

        if session.run_over:
            restarts += 1
            _control(session, "new_game")
            warmup_remaining = warmup

    ceiling_fps = float(session.speed_ceiling)
    budget_ms = 1000.0 / ceiling_fps
    unpaced_overruns = sum(1 for t in full_tick_samples if t >= budget_ms)

    # Paced probe at the clamped ceiling (set_speed(120) -> clamps to ceiling_fps)
    _control(session, "set_speed", 120)
    clamped_speed = float(session.speed)
    if clamped_speed != ceiling_fps:
        raise RuntimeError(f"set_speed(120) clamped to {clamped_speed}, expected {ceiling_fps}")
    _control(session, "new_game")
    warmup_rem = warmup
    while warmup_rem > 0:
        _full_play_tick(session, stand_in)
        warmup_rem -= 1
        if session.run_over:
            _control(session, "new_game")
            warmup_rem = warmup

    paced_ticks_ms: List[float] = []
    paced_start = time.perf_counter()
    for _ in range(PACED_TICKS):
        _, _, ft_ms, _ = _full_play_tick(session, stand_in)
        paced_ticks_ms.append(ft_ms)
        if session.run_over:
            _control(session, "new_game")
            w_rem = warmup
            while w_rem > 0:
                _full_play_tick(session, stand_in)
                w_rem -= 1
                if session.run_over:
                    _control(session, "new_game")
                    w_rem = warmup
        time.sleep(1.0 / clamped_speed)
    paced_elapsed = time.perf_counter() - paced_start
    paced_overruns = sum(1 for t in paced_ticks_ms if t >= budget_ms)
    session.close(wait=True)

    return {
        "roster": roster,
        "opponents": opponents,
        "seed": seed,
        "warmup_frames_per_run": warmup,
        "measured_ticks": len(full_tick_samples),
        "total_warmup_frames": warmup_frames_total,
        "restarts": restarts,
        "ceiling_fps": ceiling_fps,
        "budget_ms": round(budget_ms, 3),
        "roster_verification": roster_info,
        "update_ms": _distribution(update_samples),
        "step_snapshot_ms": _distribution(step_snapshot_samples),
        "full_tick_ms": _distribution(full_tick_samples),
        "input_ms": _distribution(input_samples),
        "unpaced_overruns": unpaced_overruns,
        "paced_ceiling_probe": {
            "requested_fps": 120,
            "clamped_fps": clamped_speed,
            "ticks": PACED_TICKS,
            "budget_ms": round(budget_ms, 3),
            "overruns": paced_overruns,
            "full_tick_ms": _distribution(paced_ticks_ms),
            "achieved_fps": PACED_TICKS / paced_elapsed,
        },
    }


def measure_play_watch_switching_rss(
    switches: int = 10, steps_per_mode: int = 30, opponents: int = 8, seed: int = 20261104
) -> Dict[str, Any]:
    """Switch Play (mixed) <-> Watch (mixed) 10 times and record RSS after each switch."""
    _seed_everything(seed)
    set_world_seed(seed)
    session = GameSession()
    stand_in = ScriptedHuman(seed)

    # Initial Watch warmup so baseline RSS includes initialized PyTorch/Numba buffers.
    for _ in range(50):
        session.step()
        session.snapshot()
    gc.collect()

    rss_before = _current_rss_bytes()
    maxrss_before = _max_rss_bytes()
    records: List[Dict[str, Any]] = []

    for idx in range(1, switches + 1):
        _control(session, "set_mode", "play")
        _control(session, "set_play_opponents", opponents)
        _control(session, "set_play_roster", "mixed")
        _verify_play_session(session, "mixed", opponents)
        for _ in range(steps_per_mode):
            _full_play_tick(session, stand_in)
            if session.run_over:
                _control(session, "new_game")
        rss_play = _current_rss_bytes()

        _control(session, "set_mode", "watch")
        if session.mode != MODE_WATCH or session.watch_roster != "mixed":
            raise RuntimeError(
                f"expected Watch mixed after switch {idx}, "
                f"got mode={session.mode} roster={session.watch_roster}"
            )
        for _ in range(steps_per_mode):
            session.step()
            session.snapshot()
        gc.collect()
        rss_watch = _current_rss_bytes()

        records.append(
            {
                "switch": idx,
                "play_rss_mib": round(rss_play / (1024 * 1024), 2),
                "watch_rss_mib": round(rss_watch / (1024 * 1024), 2),
                "maxrss_mib": round(_max_rss_bytes() / (1024 * 1024), 2),
            }
        )

    session.close(wait=True)
    gc.collect()
    rss_after = _current_rss_bytes()
    maxrss_after = _max_rss_bytes()
    post_warmup_watch_mib = records[0]["watch_rss_mib"] if records else rss_before / (1024 * 1024)
    final_watch_mib = records[-1]["watch_rss_mib"] if records else rss_after / (1024 * 1024)

    return {
        "switches": switches,
        "steps_per_mode": steps_per_mode,
        "play_opponents": opponents,
        "rss_before_mib": round(rss_before / (1024 * 1024), 2),
        "rss_after_mib": round(rss_after / (1024 * 1024), 2),
        "rss_delta_start_to_end_mib": round((rss_after - rss_before) / (1024 * 1024), 2),
        "rss_delta_switch1_to_switch10_mib": round(final_watch_mib - post_warmup_watch_mib, 2),
        "peak_maxrss_before_mib": round(maxrss_before / (1024 * 1024), 2),
        "peak_maxrss_after_mib": round(maxrss_after / (1024 * 1024), 2),
        "history": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/Users/josenunez/Projects/ml/snake-dqn-artifacts/"
            "t34-sol/argon-play-mixed-build/QUALIFICATION.json"
        ),
    )
    args = parser.parse_args()

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="play-mixed-qual-") as temp_dir:
        with _isolated_serving_env(temp_dir):
            cases: List[Dict[str, Any]] = []
            base_seed = 2026110400
            case_idx = 0
            for roster in ROSTERS:
                for n in OPPONENT_COUNTS:
                    case_idx += 1
                    seed = base_seed + case_idx
                    print(
                        f"[{case_idx}/10] Qualifying Play roster={roster} N={n} "
                        f"(warmup={WARMUP_FRAMES}, ticks={MEASURED_TICKS})...",
                        flush=True,
                    )
                    res = measure_play_case(roster, n, seed)
                    cases.append(res)
                    print(
                        f"  -> ceiling={res['ceiling_fps']:.0f}fps "
                        f"update p95={res['update_ms']['p95']:.2f}ms "
                        f"p99={res['update_ms']['p99']:.2f}ms | "
                        f"full_tick p95={res['full_tick_ms']['p95']:.2f}ms "
                        f"p99={res['full_tick_ms']['p99']:.2f}ms "
                        f"max={res['full_tick_ms']['max']:.2f}ms | "
                        f"overruns={res['unpaced_overruns']}/{res['measured_ticks']} "
                        f"(paced={res['paced_ceiling_probe']['overruns']}/{PACED_TICKS})",
                        flush=True,
                    )

            print("[RSS] Running Play <-> Watch switching RSS check (10 switches)...", flush=True)
            rss_check = measure_play_watch_switching_rss(switches=10)
            print(
                f"  -> RSS before={rss_check['rss_before_mib']:.2f} MiB, "
                f"after={rss_check['rss_after_mib']:.2f} MiB, "
                f"delta(switch1->10)={rss_check['rss_delta_switch1_to_switch10_mib']:+.2f} MiB",
                flush=True,
            )

    payload = {
        "schema": "play-mixed-mac-serving-qualification/v1",
        "created_utc": _utc_now(),
        "wall_seconds": round(time.monotonic() - started, 2),
        "protocol": {
            "warmup_frames": WARMUP_FRAMES,
            "measured_ticks": MEASURED_TICKS,
            "paced_ticks": PACED_TICKS,
            "opponent_counts": list(OPPONENT_COUNTS),
            "rosters": list(ROSTERS),
            "ceilings": {"ai_le_6_fps": 24.0, "ai_gt_6_fps": 12.0},
            "threading": {
                "torch_num_threads": torch.get_num_threads(),
                "device": os.environ.get("SNAKE_DQN_DEVICE"),
            },
        },
        "cases": cases,
        "rss_switching": rss_check,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
