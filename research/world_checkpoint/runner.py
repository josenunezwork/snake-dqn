"""Resumable paired-world runner with periodic and SIGUSR1/SIGTERM checkpointing."""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.engine import (
    COLLAPSE_DECISIONS,
    COLLAPSE_SHARE,
    ScientificFault,
)
from research.watch6_l1_confirm.run import DEFAULT_LOCAL_SAVED_DIR, WATCHDOG_SECONDS
from research.watch6_search_efficacy.engine import DenseTrigger, dense_batch, step_guard
from research.watch6_search_efficacy.product import open_watch
from research.world_checkpoint.checkpoint import (
    WORLD_CHECKPOINT_SCHEMA,
    CheckpointIdentityError,
    capture_runtime_identity,
    configure_cpu,
    find_latest_checkpoint,
    highest_checkpoint_seq_on_disk,
    load_checkpoint,
    quarantine_checkpoint_file,
    quarantine_incomplete_checkpoints,
    restore_run_session_state,
    restore_trigger_state,
    save_checkpoint,
    snapshot_run_session_state,
    snapshot_trigger_state,
    state_digest,
    verify_checkpoint_identity,
    verify_session_parity,
)
from src.simd_env.batch_sim import CARDINAL
from src.simd_env.grid_scenarios import inject_serpentines
from tools.ops.package_spec import digest, publish_json

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_INTERVAL = 1800
GRACE_SECONDS = 15.0
WORLD_UNIT_SCHEMA = "watch6-l1-world/v1"

_EXTERNAL_SIGNAL_GETTER: Callable[[], int | None] | None = None


def set_external_signal_getter(getter: Callable[[], int | None] | None) -> None:
    """Register a process-level signal polling callback (used by worker pools)."""
    global _EXTERNAL_SIGNAL_GETTER
    _EXTERNAL_SIGNAL_GETTER = getter


class CheckpointInterrupted(RuntimeError):
    """Raised when a paired-world run saves a verified checkpoint and stops early."""

    def __init__(
        self,
        message: str,
        *,
        checkpoint_path: Path,
        arm: str | None,
        frame_index: int,
        signal_number: int | None = None,
        reason: str = "interrupted",
        ignored_checkpoints: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.checkpoint_path = Path(checkpoint_path)
        self.arm = arm
        self.frame_index = int(frame_index)
        self.signal_number = signal_number
        self.reason = str(reason)
        self.ignored_checkpoints = [dict(x) for x in ignored_checkpoints]


class _SignalAbortSearch(Exception):
    """Internal exception raised inside step_guard when a signal interrupts dense_batch."""


def _compute_session_roster(session: Any) -> dict[str, Any]:
    """Compute the authoritative model/mechanics roster from a live Watch6 session."""
    from research.t15_mixed_roster_20261017.episodes import roster_identity
    from src.core.game_config import GameConfig

    live_roster = roster_identity(session.live)
    return {
        "hero_checkpoint_sha256": digest(Path(session.live.served_checkpoint["path"])),
        "opponent_checkpoint_sha256": digest(Path(session.live.watch_opponent_checkpoint["path"])),
        "opponent_count": 5,
        "veto_variant": "v8",
        "veto_lambda": 8.0,
        "mechanics_version": int(GameConfig.MECHANICS_VERSION),
        "num_snakes": int(GameConfig.NUM_SNAKES),
        "live_roster": live_roster,
    }


def _pack_arm_counters(
    *,
    arm: str,
    seed: int,
    horizon: int,
    frame_index: int,
    min_length: int | None,
    interval: int,
    trigger: DenseTrigger | None,
    was_alive: bool,
    previous_alive: Sequence[bool],
    respawns: Sequence[int],
    roster: Mapping[str, Any],
    decisions: int,
    boost: int,
    deaths: int,
    alive_frames: int,
    mass_sum: int,
    batches: int,
    batches_lt600: int,
    batches_ge600: int,
    batches_ge1000: int,
    interventions: int,
    interventions_lt600: int,
    interventions_ge600: int,
    evaluated: int,
    branches_lt600: int,
    branches_600_999: int,
    branches_ge1000: int,
    skipped: int,
    kernel_rows: int,
    mirror_frames: int,
    main_seconds: float,
    search_seconds: float,
    search_seconds_lt600: float,
    search_seconds_600_999: float,
    search_seconds_ge1000: float,
    action_counts: Sequence[int],
    recent_actions: Sequence[int],
    all_frame_masses: Sequence[int],
    alive_lengths: Sequence[int],
    death_lengths: Sequence[int],
) -> dict[str, Any]:
    return {
        "arm": str(arm),
        "seed": int(seed),
        "horizon": int(horizon),
        "frame_index": int(frame_index),
        "min_length": None if min_length is None else int(min_length),
        "interval": int(interval),
        "trigger": snapshot_trigger_state(trigger),
        "was_alive": bool(was_alive),
        "previous_alive": [bool(x) for x in previous_alive],
        "respawns": [int(x) for x in respawns],
        "roster": dict(roster),
        "decisions": int(decisions),
        "boost": int(boost),
        "deaths": int(deaths),
        "alive_frames": int(alive_frames),
        "mass_sum": int(mass_sum),
        "batches": int(batches),
        "batches_lt600": int(batches_lt600),
        "batches_ge600": int(batches_ge600),
        "batches_ge1000": int(batches_ge1000),
        "interventions": int(interventions),
        "interventions_lt600": int(interventions_lt600),
        "interventions_ge600": int(interventions_ge600),
        "evaluated": int(evaluated),
        "branches_lt600": int(branches_lt600),
        "branches_600_999": int(branches_600_999),
        "branches_ge1000": int(branches_ge1000),
        "skipped": int(skipped),
        "kernel_rows": int(kernel_rows),
        "mirror_frames": int(mirror_frames),
        "main_seconds": float(main_seconds),
        "search_seconds": float(search_seconds),
        "search_seconds_lt600": float(search_seconds_lt600),
        "search_seconds_600_999": float(search_seconds_600_999),
        "search_seconds_ge1000": float(search_seconds_ge1000),
        "action_counts": [int(x) for x in action_counts],
        "recent_actions": [int(x) for x in recent_actions],
        "all_frame_masses": [int(x) for x in all_frame_masses],
        "alive_lengths": [int(x) for x in alive_lengths],
        "death_lengths": [int(x) for x in death_lengths],
    }


def _will_trigger_fire_this_frame(
    trigger: DenseTrigger | None,
    alive: bool,
    length: int,
    decisions: int,
    *,
    min_length: int | None = None,
    was_alive: bool = True,
) -> bool:
    """Predict without side effects whether DenseTrigger could fire on the upcoming frame.

    Includes respawn-eligible frames (``not was_alive`` or ``not alive`` or ``trigger is None``)
    because ``step_with_policy`` respawns dead snakes before calling ``choose``, which creates
    a fresh ``DenseTrigger`` whose ``first_decision`` is ``None``.
    """
    if min_length is not None and not isinstance(min_length, int):
        return False
    if not was_alive or not alive or trigger is None:
        return isinstance(min_length, int) or trigger is not None
    if int(length) < int(trigger.min_length):
        return False
    if trigger.first_decision is None:
        return True
    return (int(decisions) - int(trigger.first_decision)) % int(trigger.interval) == 0


def _build_world_checkpoint_payload(
    *,
    checkpoint_seq: int,
    index: int,
    seed: int,
    horizon: int,
    arm_order: Sequence[str],
    fixture: bool,
    completed_arms: Mapping[str, Any],
    rosters: Sequence[Mapping[str, Any]],
    elapsed_seconds_accum: float,
    current_arm: str | None,
    current_frame: int,
    in_progress_arm: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "checkpoint_schema": WORLD_CHECKPOINT_SCHEMA,
        "checkpoint_seq": int(checkpoint_seq),
        "world_schema": WORLD_UNIT_SCHEMA,
        "identity": capture_runtime_identity(),
        "index": int(index),
        "seed": int(seed),
        "horizon": int(horizon),
        "arm_order": [str(a) for a in arm_order],
        "fixture": bool(fixture),
        "completed_arms": {str(k): dict(v) for k, v in completed_arms.items()},
        "rosters": [dict(r) for r in rosters],
        "elapsed_seconds_accum": float(elapsed_seconds_accum),
        "current_arm": None if current_arm is None else str(current_arm),
        "current_frame": int(current_frame),
        "in_progress_arm": None if in_progress_arm is None else dict(in_progress_arm),
    }


def _validate_checkpoint_matches_world(
    ckpt: Mapping[str, Any],
    *,
    index: int,
    seed: int,
    horizon: int,
    arm_order: Sequence[str],
    fixture: bool,
) -> None:
    verify_checkpoint_identity(ckpt)
    expected = {
        "world_schema": WORLD_UNIT_SCHEMA,
        "index": int(index),
        "seed": int(seed),
        "horizon": int(horizon),
        "arm_order": [str(a) for a in arm_order],
        "fixture": bool(fixture),
    }
    for key, val in expected.items():
        actual = ckpt.get(key)
        if actual != val:
            raise CheckpointIdentityError(
                f"checkpoint {key}={actual!r} does not match requested {val!r}"
            )


def run_paired_world_resumable(
    index: int,
    seed: int,
    arm_order: Sequence[str],
    check: Callable[[], None] | None = None,
    *,
    horizon: int = p.HORIZON,
    fixture: bool = False,
    checkpoint_dir: str | Path | None = None,
    resume_checkpoint_dirs: Sequence[str | Path] | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    install_signal_handlers: bool = True,
    watchdog_seconds: int | None = WATCHDOG_SECONDS,
    on_progress: Callable[[str, int], None] | None = None,
    on_frame_digest: Callable[[str, int, str], None] | None = None,
    stop_at: Callable[[str, int], bool] | tuple[str, int] | None = None,
    signal_on_search: tuple[str, int, int] | None = None,
    initial_session_hook: Callable[[Any, str], None] | None = None,
    external_signal_getter: Callable[[], int | None] | None = None,
) -> dict[str, Any]:
    """Run a Watch6-L1-V1 paired world with periodic/signal checkpoints and verified resume."""
    os.environ.setdefault("SNAKE_WATCH_SOURCE", "live")
    if check is None:
        check = lambda: None  # noqa: E731
    if not fixture and (type(index) is not int or not 1 <= index <= p.COUNT):
        raise ValueError(f"production world index must be in 1..{p.COUNT}, got {index!r}")
    if type(horizon) is not int or not 1 <= horizon <= p.HORIZON:
        raise ValueError(f"horizon must be an int in 1..{p.HORIZON}, got {horizon!r}")
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError(f"checkpoint_interval must be a positive int, got {checkpoint_interval!r}")

    order = tuple(str(arm) for arm in arm_order)
    if not order or any(arm not in p.ARM_SPECS for arm in order) or len(set(order)) != len(order):
        raise ValueError(f"arm_order must be distinct arms from {p.ARMS}, got {arm_order!r}")
    if not fixture and (sorted(order) != sorted(p.ARMS) or len(order) != 3):
        raise ValueError(f"arm_order must be a permutation of {p.ARMS}, got {arm_order!r}")

    ckpt_root = None if checkpoint_dir is None else Path(checkpoint_dir)
    extra_resume_dirs = [Path(d) for d in (resume_checkpoint_dirs or ())]
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)
        quarantine_incomplete_checkpoints(ckpt_root)

    stop_predicate: Callable[[str, int], bool] | None = None
    if isinstance(stop_at, tuple):
        stop_arm, stop_frame = str(stop_at[0]), int(stop_at[1])
        stop_predicate = lambda a, f: a == stop_arm and f == stop_frame  # noqa: E731
    elif callable(stop_at):
        stop_predicate = stop_at

    # Attempt to resume from the newest verified checkpoint across ckpt_root and extra_resume_dirs.
    completed_arms: dict[str, dict[str, Any]] = {}
    rosters: list[dict[str, Any]] = []
    elapsed_seconds_accum = 0.0
    checkpoint_seq = highest_checkpoint_seq_on_disk(ckpt_root, *extra_resume_dirs)
    resumed_in_progress: dict[str, Any] | None = None
    ignored_checkpoints: list[dict[str, Any]] = []
    loaded_ckpt_path: Path | None = None
    loaded_ckpt_data: dict[str, Any] | None = None

    def _record_and_ignore_mismatched_checkpoint(
        ckpt_path: Path,
        ckpt_data: Mapping[str, Any] | None,
        reason: str,
    ) -> None:
        nonlocal elapsed_seconds_accum, resumed_in_progress, checkpoint_seq
        seq_num = int(ckpt_data.get("checkpoint_seq", 0)) if isinstance(ckpt_data, Mapping) else 0
        checkpoint_seq = max(
            checkpoint_seq,
            seq_num,
            highest_checkpoint_seq_on_disk(ckpt_root, *extra_resume_dirs),
        )
        record = {
            "index": int(index),
            "seed": int(seed),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_seq": seq_num,
            "current_arm": (
                ckpt_data.get("current_arm") if isinstance(ckpt_data, Mapping) else None
            ),
            "current_frame": (
                int(ckpt_data.get("current_frame", 0)) if isinstance(ckpt_data, Mapping) else 0
            ),
            "reason": str(reason),
            "action": "ignored_start_fresh",
        }
        ignored_checkpoints.append(record)
        logger.warning(
            "Refused mismatched checkpoint %s (%s); ignoring and starting world %s fresh",
            ckpt_path,
            reason,
            index,
        )
        # Never mix states: reset progress and quarantine mismatched checkpoints in ckpt_root
        completed_arms.clear()
        rosters.clear()
        elapsed_seconds_accum = 0.0
        resumed_in_progress = None
        if ckpt_root is not None and ckpt_root.is_dir():
            for existing_ckpt in sorted(ckpt_root.glob("*.ckpt")):
                quarantine_checkpoint_file(
                    existing_ckpt, f"ignored mismatched checkpoint: {reason}"
                )
            marker_seq = max(1, checkpoint_seq)
            marker_path = ckpt_root / f"ignored-checkpoints-seq-{marker_seq:06d}.json"
            suffix = 1
            while marker_path.exists():
                marker_path = ckpt_root / f"ignored-checkpoints-seq-{marker_seq:06d}-{suffix}.json"
                suffix += 1
            publish_json(
                marker_path,
                {
                    "schema": "watch6-world-checkpoint-ignored/v1",
                    "index": int(index),
                    "seed": int(seed),
                    "horizon": int(horizon),
                    "fixture": bool(fixture),
                    "ignored_checkpoints": list(ignored_checkpoints),
                },
            )

    if ckpt_root is not None or extra_resume_dirs:
        latest_path = find_latest_checkpoint(
            ckpt_root,
            extra_dirs=extra_resume_dirs,
            verify_identity=False,
        )
        if latest_path is not None:
            raw_ckpt_data = load_checkpoint(latest_path, verify_identity=False)
            try:
                _validate_checkpoint_matches_world(
                    raw_ckpt_data,
                    index=index,
                    seed=seed,
                    horizon=horizon,
                    arm_order=order,
                    fixture=fixture,
                )
            except (CheckpointIdentityError, ValueError) as exc:
                _record_and_ignore_mismatched_checkpoint(latest_path, raw_ckpt_data, str(exc))
            else:
                loaded_ckpt_path = latest_path
                loaded_ckpt_data = raw_ckpt_data
                completed_arms = {
                    str(k): dict(v) for k, v in raw_ckpt_data["completed_arms"].items()
                }
                rosters = [dict(r) for r in raw_ckpt_data["rosters"]]
                elapsed_seconds_accum = float(raw_ckpt_data.get("elapsed_seconds_accum", 0.0))
                checkpoint_seq = max(checkpoint_seq, int(raw_ckpt_data.get("checkpoint_seq", 0)))
                if raw_ckpt_data.get("in_progress_arm") is not None:
                    resumed_in_progress = dict(raw_ckpt_data["in_progress_arm"])

    signal_state: dict[str, int | float | None] = {
        "signum": None,
        "received_at": None,
    }
    previous_handlers: dict[int, Any] = {}
    can_install_signals = (
        bool(install_signal_handlers) and threading.current_thread() is threading.main_thread()
    )

    def _on_signal(signum: int, _frame: Any) -> None:
        if signal_state["signum"] is None:
            signal_state["signum"] = int(signum)
            signal_state["received_at"] = time.monotonic()

    def _poll_external_signal() -> None:
        if signal_state["signum"] is not None:
            return
        getter = external_signal_getter or _EXTERNAL_SIGNAL_GETTER
        if getter is not None:
            ext_sig = getter()
            if ext_sig is not None and int(ext_sig) > 0:
                signal_state["signum"] = int(ext_sig)
                signal_state["received_at"] = time.monotonic()

    if can_install_signals:
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _on_signal)
    _poll_external_signal()

    def _write_ckpt(
        *,
        current_arm: str | None,
        current_frame: int,
        in_progress_arm: Mapping[str, Any] | None,
        run_started_mono: float,
    ) -> Path | None:
        nonlocal checkpoint_seq
        if ckpt_root is None:
            return None
        checkpoint_seq = (
            max(checkpoint_seq, highest_checkpoint_seq_on_disk(ckpt_root, *extra_resume_dirs)) + 1
        )
        arm_tag = current_arm if current_arm is not None else "boundary"
        ckpt_name = f"ckpt-seq-{checkpoint_seq:06d}-arm-{arm_tag}-frame-{current_frame:05d}.ckpt"
        ckpt_path = ckpt_root / ckpt_name
        payload = _build_world_checkpoint_payload(
            checkpoint_seq=checkpoint_seq,
            index=index,
            seed=seed,
            horizon=horizon,
            arm_order=order,
            fixture=fixture,
            completed_arms=completed_arms,
            rosters=rosters,
            elapsed_seconds_accum=elapsed_seconds_accum + (time.monotonic() - run_started_mono),
            current_arm=current_arm,
            current_frame=current_frame,
            in_progress_arm=in_progress_arm,
        )
        save_checkpoint(payload, ckpt_path)
        return ckpt_path

    from research.bl_served_search.ceiling import _legal
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q
    from research.t15_mixed_roster_20261017.episodes import _assert_mixed
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.eval_engine import _dispatch_actions

    use_watchdog = watchdog_seconds is not None and int(watchdog_seconds) > 0
    if use_watchdog:
        faulthandler.dump_traceback_later(int(watchdog_seconds), exit=True)
    last_watchdog = time.monotonic()

    run_started_mono = time.monotonic()
    verified_fresh_roster_once = False
    try:
        for _roster_attempt in range(2):
            restarted_due_to_roster = False
            for arm in order:
                if arm in completed_arms:
                    continue

                if use_watchdog:
                    faulthandler.dump_traceback_later(int(watchdog_seconds), exit=True)
                    last_watchdog = time.monotonic()

                spec = p.ARM_SPECS[arm]
                min_length = spec["min_length"]
                interval = int(spec["interval"])

                check()
                _poll_external_signal()
                session = open_watch({"seed": int(seed)}, 24, "bare")
                try:
                    world = session.world
                    if not world.sim.allow_respawn or any(
                        not getattr(s, "auto_respawn", True) for s in session.live.game.snakes
                    ):
                        raise ScientificFault(
                            "Watch6-L1-V1 requires continuous respawn in live and mirror"
                        )
                    if getattr(session.live, "pathb", None) is not None:
                        raise ScientificFault("Watch6-L1-V1 refuses Path B assistance")
                    _assert_mixed(session.live, 6)

                    # Always recompute roster from loaded models and verify against checkpoint.
                    fresh_roster = _compute_session_roster(session)
                    if not verified_fresh_roster_once and loaded_ckpt_path is not None:
                        in_prog_roster = (
                            dict(resumed_in_progress["counters"]["roster"])
                            if (
                                resumed_in_progress is not None
                                and isinstance(resumed_in_progress.get("counters"), Mapping)
                                and isinstance(
                                    resumed_in_progress["counters"].get("roster"), Mapping
                                )
                            )
                            else None
                        )
                        if any(r != fresh_roster for r in rosters) or (
                            in_prog_roster is not None and in_prog_roster != fresh_roster
                        ):
                            mismatch_msg = (
                                "checkpoint completed-arm roster does not match "
                                "newly loaded model roster"
                                if any(r != fresh_roster for r in rosters)
                                else (
                                    "checkpoint in-progress roster does not match "
                                    "newly loaded model roster"
                                )
                            )
                            _record_and_ignore_mismatched_checkpoint(
                                loaded_ckpt_path,
                                loaded_ckpt_data,
                                mismatch_msg,
                            )
                            loaded_ckpt_path = None
                            loaded_ckpt_data = None
                            verified_fresh_roster_once = True
                            restarted_due_to_roster = True
                            break
                    verified_fresh_roster_once = True

                    if resumed_in_progress is not None and resumed_in_progress.get("arm") == arm:
                        arm_counters = dict(resumed_in_progress["counters"])
                        restore_run_session_state(session, resumed_in_progress["session"])
                        start_frame = int(arm_counters["frame_index"]) + 1
                        trigger = restore_trigger_state(arm_counters["trigger"])
                        was_alive = bool(arm_counters["was_alive"])
                        previous_alive = [bool(x) for x in arm_counters["previous_alive"]]
                        respawns = [int(x) for x in arm_counters["respawns"]]
                        roster = fresh_roster
                        decisions = int(arm_counters["decisions"])
                        boost = int(arm_counters["boost"])
                        deaths = int(arm_counters["deaths"])
                        alive_frames = int(arm_counters["alive_frames"])
                        mass_sum = int(arm_counters["mass_sum"])
                        batches = int(arm_counters["batches"])
                        batches_lt600 = int(arm_counters["batches_lt600"])
                        batches_ge600 = int(arm_counters["batches_ge600"])
                        batches_ge1000 = int(arm_counters["batches_ge1000"])
                        interventions = int(arm_counters["interventions"])
                        interventions_lt600 = int(arm_counters["interventions_lt600"])
                        interventions_ge600 = int(arm_counters["interventions_ge600"])
                        evaluated = int(arm_counters["evaluated"])
                        branches_lt600 = int(arm_counters["branches_lt600"])
                        branches_600_999 = int(arm_counters["branches_600_999"])
                        branches_ge1000 = int(arm_counters["branches_ge1000"])
                        skipped = int(arm_counters["skipped"])
                        kernel_rows = int(arm_counters["kernel_rows"])
                        mirror_frames = int(arm_counters["mirror_frames"])
                        main_seconds = float(arm_counters["main_seconds"])
                        search_seconds = float(arm_counters["search_seconds"])
                        search_seconds_lt600 = float(arm_counters["search_seconds_lt600"])
                        search_seconds_600_999 = float(arm_counters["search_seconds_600_999"])
                        search_seconds_ge1000 = float(arm_counters["search_seconds_ge1000"])
                        action_counts = [int(x) for x in arm_counters["action_counts"]]
                        recent_actions = [int(x) for x in arm_counters["recent_actions"]]
                        all_frame_masses = [int(x) for x in arm_counters["all_frame_masses"]]
                        alive_lengths = [int(x) for x in arm_counters["alive_lengths"]]
                        death_lengths = [int(x) for x in arm_counters["death_lengths"]]
                        resumed_in_progress = None
                    else:
                        if initial_session_hook is not None:
                            initial_session_hook(session, arm)
                        start_frame = 1
                        roster = fresh_roster
                        trigger = (
                            DenseTrigger(min_length=int(min_length), interval=interval)
                            if isinstance(min_length, int)
                            else None
                        )
                        was_alive = bool(world.sim.alive[0, 0])
                        previous_alive = [bool(s.is_alive) for s in session.live.game.snakes]
                        respawns = [0] * 6
                        decisions = 0
                        boost = 0
                        deaths = 0
                        alive_frames = 0
                        mass_sum = 0
                        batches = 0
                        batches_lt600 = 0
                        batches_ge600 = 0
                        batches_ge1000 = 0
                        interventions = 0
                        interventions_lt600 = 0
                        interventions_ge600 = 0
                        evaluated = 0
                        branches_lt600 = 0
                        branches_600_999 = 0
                        branches_ge1000 = 0
                        skipped = 0
                        kernel_rows = 0
                        mirror_frames = 0
                        main_seconds = 0.0
                        search_seconds = 0.0
                        search_seconds_lt600 = 0.0
                        search_seconds_600_999 = 0.0
                        search_seconds_ge1000 = 0.0
                        action_counts = [0] * 6
                        recent_actions = []
                        all_frame_masses = []
                        alive_lengths = []
                        death_lengths = []

                    def _current_in_progress_dict(frame_idx: int) -> dict[str, Any]:
                        counters = _pack_arm_counters(
                            arm=arm,
                            seed=seed,
                            horizon=horizon,
                            frame_index=frame_idx,
                            min_length=min_length,
                            interval=interval,
                            trigger=trigger,
                            was_alive=was_alive,
                            previous_alive=previous_alive,
                            respawns=respawns,
                            roster=roster,
                            decisions=decisions,
                            boost=boost,
                            deaths=deaths,
                            alive_frames=alive_frames,
                            mass_sum=mass_sum,
                            batches=batches,
                            batches_lt600=batches_lt600,
                            batches_ge600=batches_ge600,
                            batches_ge1000=batches_ge1000,
                            interventions=interventions,
                            interventions_lt600=interventions_lt600,
                            interventions_ge600=interventions_ge600,
                            evaluated=evaluated,
                            branches_lt600=branches_lt600,
                            branches_600_999=branches_600_999,
                            branches_ge1000=branches_ge1000,
                            skipped=skipped,
                            kernel_rows=kernel_rows,
                            mirror_frames=mirror_frames,
                            main_seconds=main_seconds,
                            search_seconds=search_seconds,
                            search_seconds_lt600=search_seconds_lt600,
                            search_seconds_600_999=search_seconds_600_999,
                            search_seconds_ge1000=search_seconds_ge1000,
                            action_counts=action_counts,
                            recent_actions=recent_actions,
                            all_frame_masses=all_frame_masses,
                            alive_lengths=alive_lengths,
                            death_lengths=death_lengths,
                        )
                        return {
                            "arm": arm,
                            "frame_index": int(frame_idx),
                            "counters": counters,
                            "session": snapshot_run_session_state(session),
                        }

                    in_search_branch = False
                    search_signal_sent = False

                    def _guarded_check() -> None:
                        nonlocal search_signal_sent, last_watchdog
                        if use_watchdog:
                            now_mono = time.monotonic()
                            if now_mono - last_watchdog >= 60.0:
                                faulthandler.dump_traceback_later(int(watchdog_seconds), exit=True)
                                last_watchdog = now_mono
                        check()
                        _poll_external_signal()
                        if (
                            in_search_branch
                            and signal_on_search is not None
                            and not search_signal_sent
                            and signal_on_search[0] == arm
                            and int(signal_on_search[1]) == current_frame_index
                        ):
                            search_signal_sent = True
                            os.kill(os.getpid(), int(signal_on_search[2]))
                        if in_search_branch and signal_state["signum"] is not None:
                            raise _SignalAbortSearch()

                    with step_guard(_guarded_check):
                        for frame_index in range(start_frame, horizon + 1):
                            current_frame_index = frame_index
                            frame_started = time.monotonic()
                            step_search = 0.0
                            _guarded_check()

                            if signal_state["signum"] is not None:
                                sig_num = int(signal_state["signum"])
                                ckpt_path = _write_ckpt(
                                    current_arm=arm,
                                    current_frame=frame_index - 1,
                                    in_progress_arm=_current_in_progress_dict(frame_index - 1),
                                    run_started_mono=run_started_mono,
                                )
                                raise CheckpointInterrupted(
                                    f"signal {sig_num} received before frame "
                                    f"{frame_index} of {arm}",
                                    checkpoint_path=ckpt_path or Path(""),
                                    arm=arm,
                                    frame_index=frame_index - 1,
                                    signal_number=sig_num,
                                    reason=f"signal:{sig_num}",
                                    ignored_checkpoints=ignored_checkpoints,
                                )

                            if on_progress is not None:
                                on_progress(arm, frame_index)

                            # Snapshot before search-eligible frames (including respawn frames
                            # where choose() creates a fresh DenseTrigger) so if SIGUSR1/SIGTERM
                            # interrupts dense_batch mid-search, we can roll back to frame_index - 1
                            # and write a verified checkpoint in milliseconds.
                            pre_search_snapshot: dict[str, Any] | None = None
                            if _will_trigger_fire_this_frame(
                                trigger,
                                bool(world.sim.alive[0, 0]),
                                int(world.sim.length[0, 0]),
                                decisions,
                                min_length=min_length,
                                was_alive=was_alive,
                            ):
                                pre_search_snapshot = _current_in_progress_dict(frame_index - 1)

                            selected: dict[str, Any] = {}

                            def choose(sim: Any) -> np.ndarray:
                                nonlocal batches, batches_lt600, batches_ge600, batches_ge1000
                                nonlocal interventions, interventions_lt600, interventions_ge600
                                nonlocal evaluated, branches_lt600
                                nonlocal branches_600_999, branches_ge1000
                                nonlocal skipped, kernel_rows, step_search, trigger
                                nonlocal search_seconds_lt600, search_seconds_600_999
                                nonlocal search_seconds_ge1000, in_search_branch
                                world.runtime.prepare(sim)
                                active = bool(sim.alive[0, 0])
                                if active and not was_alive and isinstance(min_length, int):
                                    trigger = DenseTrigger(
                                        min_length=int(min_length), interval=interval
                                    )
                                actions = np.ones((1, sim.S), dtype=np.int64)
                                if active:
                                    length = int(sim.length[0, 0])
                                    obs = build_ego_raster(
                                        sim,
                                        EgoRasterConfig(version="bv8"),
                                        backend="numba",
                                        rows=np.array([[0, 0]]),
                                    )
                                    q_row = np.asarray(prepared_q(world, obs), dtype=np.float32)
                                    if not np.isfinite(q_row).all():
                                        raise ScientificFault("nonfinite hero Q row")
                                    advisory = sim.get_advisory_action_mask()[0, 0]
                                    legal, bare_action = _legal(q_row, advisory)
                                    chosen = int(bare_action)
                                    fired = trigger is not None and trigger.fires(
                                        decisions, length, True
                                    )
                                    if fired and bool(advisory.any()):
                                        t0 = time.monotonic()
                                        in_search_branch = True
                                        try:
                                            batch = dense_batch(
                                                world, frame_index - 1, q_row, legal
                                            )
                                        finally:
                                            in_search_branch = False
                                        batch_dt = time.monotonic() - t0
                                        step_search += batch_dt
                                        chosen = int(batch["chosen"])
                                        if not 0 <= chosen < 6 or not bool(legal[chosen]):
                                            raise ScientificFault(
                                                "illegal exact-stop action selection"
                                            )
                                        batches += 1
                                        intervened = int(chosen != bare_action)
                                        interventions += intervened
                                        b_eval = int(batch["branches_evaluated"])
                                        if length < 600:
                                            batches_lt600 += 1
                                            interventions_lt600 += intervened
                                            branches_lt600 += b_eval
                                            search_seconds_lt600 += batch_dt
                                        else:
                                            batches_ge600 += 1
                                            interventions_ge600 += intervened
                                            if length < 1000:
                                                branches_600_999 += b_eval
                                                search_seconds_600_999 += batch_dt
                                            else:
                                                batches_ge1000 += 1
                                                branches_ge1000 += b_eval
                                                search_seconds_ge1000 += batch_dt
                                        evaluated += b_eval
                                        skipped += int(batch["branches_skipped"])
                                        kernel_rows += int(batch["native_v8"]["kernel_rows"])
                                    selected.update(
                                        length=length,
                                        bare=int(bare_action),
                                        chosen=chosen,
                                        q=q_row.copy(),
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
                                        raise ScientificFault("bare dispatch/Q mismatch")
                                    actions[0, 0] = selected["chosen"]
                                return actions

                            actions_holder: dict[str, np.ndarray] = {}

                            def choose_record(sim: Any) -> np.ndarray:
                                actions = choose(sim)
                                actions_holder["actions"] = actions.copy()
                                return actions

                            try:
                                world.sim.step_with_policy(choose_record)
                            except _SignalAbortSearch:
                                if pre_search_snapshot is None:
                                    raise RuntimeError(
                                        "missing pre-search snapshot on signal abort"
                                    )
                                restore_run_session_state(session, pre_search_snapshot["session"])
                                sig_num = int(signal_state["signum"] or signal.SIGUSR1)
                                ckpt_path = _write_ckpt(
                                    current_arm=arm,
                                    current_frame=frame_index - 1,
                                    in_progress_arm=pre_search_snapshot,
                                    run_started_mono=run_started_mono,
                                )
                                raise CheckpointInterrupted(
                                    f"signal {sig_num} aborted search at frame {frame_index} "
                                    f"of {arm}; checkpointed frame {frame_index - 1}",
                                    checkpoint_path=ckpt_path or Path(""),
                                    arm=arm,
                                    frame_index=frame_index - 1,
                                    signal_number=sig_num,
                                    reason=f"signal_mid_search:{sig_num}",
                                    ignored_checkpoints=ignored_checkpoints,
                                )

                            world.runtime.observe_step(world.sim)

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
                                    raise ScientificFault(
                                        f"live/mirror B8 Q mismatch at frame {frame_index}"
                                    )
                                return value

                            hero.update = observed_update
                            session.live.policy._next_dispatch_q = observed_q
                            try:
                                session._step_live(actions_holder["actions"])
                            finally:
                                hero.update = original_update
                                session.live.policy._next_dispatch_q = original_q

                            if actual_decisions != ([selected["length"]] if selected else []):
                                raise ScientificFault(
                                    "live/mirror actual hero decision mismatch "
                                    f"at frame {frame_index}"
                                )

                            hero = session.live.game.snakes[0]
                            death = int(hero.id) in session.live.game.frame_death_causes
                            grid_death = bool(
                                world.sim.get_transition_valid()[0, 0] and not world.sim.alive[0, 0]
                            )
                            if death != grid_death:
                                raise ScientificFault(
                                    f"Watch/mirror death event mismatch at frame {frame_index}"
                                )
                            if selected:
                                if int(hero.last_action) != selected["chosen"]:
                                    raise ScientificFault(
                                        f"live delivered action mismatch at frame {frame_index}"
                                    )
                                decisions += 1
                                chosen_act = int(selected["chosen"])
                                boost += int(chosen_act >= 3)
                                action_counts[chosen_act] += 1
                                recent_actions.append(chosen_act)
                                if len(recent_actions) >= COLLAPSE_DECISIONS:
                                    window = recent_actions[-COLLAPSE_DECISIONS:]
                                    if max(window.count(a) for a in range(6)) > int(
                                        COLLAPSE_DECISIONS * COLLAPSE_SHARE
                                    ):
                                        raise ScientificFault(
                                            "unconditional action-collapse fault (>90%)"
                                        )

                            was_alive = bool(hero.is_alive)
                            if was_alive != bool(world.sim.alive[0, 0]) or (
                                was_alive and int(hero.length) != int(world.sim.length[0, 0])
                            ):
                                raise ScientificFault(
                                    f"live/mirror alive or length mismatch at frame {frame_index}"
                                )

                            alive_now = [bool(s.is_alive) for s in session.live.game.snakes]
                            for slot in range(6):
                                respawns[slot] += int(alive_now[slot] and not previous_alive[slot])
                            previous_alive = alive_now

                            if death:
                                deaths += 1
                                death_lengths.append(int(selected.get("length", 0)))

                            if was_alive:
                                cur_len = int(hero.length)
                                alive_frames += 1
                                mass_sum += cur_len
                                all_frame_masses.append(cur_len)
                                alive_lengths.append(cur_len)
                            else:
                                all_frame_masses.append(0)

                            main_seconds += time.monotonic() - frame_started - step_search
                            search_seconds += step_search
                            mirror_frames += 1

                            if on_frame_digest is not None:
                                counters_for_digest = _pack_arm_counters(
                                    arm=arm,
                                    seed=seed,
                                    horizon=horizon,
                                    frame_index=frame_index,
                                    min_length=min_length,
                                    interval=interval,
                                    trigger=trigger,
                                    was_alive=was_alive,
                                    previous_alive=previous_alive,
                                    respawns=respawns,
                                    roster=roster,
                                    decisions=decisions,
                                    boost=boost,
                                    deaths=deaths,
                                    alive_frames=alive_frames,
                                    mass_sum=mass_sum,
                                    batches=batches,
                                    batches_lt600=batches_lt600,
                                    batches_ge600=batches_ge600,
                                    batches_ge1000=batches_ge1000,
                                    interventions=interventions,
                                    interventions_lt600=interventions_lt600,
                                    interventions_ge600=interventions_ge600,
                                    evaluated=evaluated,
                                    branches_lt600=branches_lt600,
                                    branches_600_999=branches_600_999,
                                    branches_ge1000=branches_ge1000,
                                    skipped=skipped,
                                    kernel_rows=kernel_rows,
                                    mirror_frames=mirror_frames,
                                    main_seconds=main_seconds,
                                    search_seconds=search_seconds,
                                    search_seconds_lt600=search_seconds_lt600,
                                    search_seconds_600_999=search_seconds_600_999,
                                    search_seconds_ge1000=search_seconds_ge1000,
                                    action_counts=action_counts,
                                    recent_actions=recent_actions,
                                    all_frame_masses=all_frame_masses,
                                    alive_lengths=alive_lengths,
                                    death_lengths=death_lengths,
                                )
                                on_frame_digest(
                                    arm, frame_index, state_digest(session, counters_for_digest)
                                )

                            # Signal check immediately after completing frame_index
                            _poll_external_signal()
                            if signal_state["signum"] is not None:
                                sig_num = int(signal_state["signum"])
                                if frame_index < horizon:
                                    ckpt_path = _write_ckpt(
                                        current_arm=arm,
                                        current_frame=frame_index,
                                        in_progress_arm=_current_in_progress_dict(frame_index),
                                        run_started_mono=run_started_mono,
                                    )
                                    raise CheckpointInterrupted(
                                        f"signal {sig_num} received at frame "
                                        f"{frame_index} of {arm}",
                                        checkpoint_path=ckpt_path or Path(""),
                                        arm=arm,
                                        frame_index=frame_index,
                                        signal_number=sig_num,
                                        reason=f"signal:{sig_num}",
                                        ignored_checkpoints=ignored_checkpoints,
                                    )

                            # Periodic mid-arm checkpoint
                            if (
                                ckpt_root is not None
                                and frame_index < horizon
                                and frame_index % checkpoint_interval == 0
                            ):
                                _write_ckpt(
                                    current_arm=arm,
                                    current_frame=frame_index,
                                    in_progress_arm=_current_in_progress_dict(frame_index),
                                    run_started_mono=run_started_mono,
                                )

                            # Deterministic stop hook for mid-arm interruption tests
                            if (
                                stop_predicate is not None
                                and frame_index < horizon
                                and stop_predicate(arm, frame_index)
                            ):
                                ckpt_path = _write_ckpt(
                                    current_arm=arm,
                                    current_frame=frame_index,
                                    in_progress_arm=_current_in_progress_dict(frame_index),
                                    run_started_mono=run_started_mono,
                                )
                                raise CheckpointInterrupted(
                                    f"stopped at requested mid-arm point ({arm}, {frame_index})",
                                    checkpoint_path=ckpt_path or Path(""),
                                    arm=arm,
                                    frame_index=frame_index,
                                    reason="stop_at",
                                    ignored_checkpoints=ignored_checkpoints,
                                )

                    if mirror_frames != horizon:
                        raise ScientificFault(
                            f"incomplete mirror frame coverage ({mirror_frames} != {horizon})"
                        )

                    arm_summary = {
                        "arm": arm,
                        "frames": horizon,
                        "alive_frames": alive_frames,
                        "mass_sum": mass_sum,
                        "deaths": deaths,
                        "decisions": decisions,
                        "boost_decisions": boost,
                        "action_counts": action_counts,
                        "length_median": float(np.median(all_frame_masses)),
                        "alive_length_median": (
                            float(np.median(alive_lengths)) if alive_lengths else 0.0
                        ),
                        "length_p95": (
                            float(np.percentile(alive_lengths, 95)) if alive_lengths else 0.0
                        ),
                        "max_length": max(alive_lengths, default=0),
                        "death_lengths": death_lengths,
                        "short_body_deaths_lt600": sum(length < 600 for length in death_lengths),
                        "long_body_deaths_ge600": sum(length >= 600 for length in death_lengths),
                        "late_large_body_deaths": sum(length >= 800 for length in death_lengths),
                        "batches": batches,
                        "batches_lt600": batches_lt600,
                        "batches_ge600": batches_ge600,
                        "batches_ge1000": batches_ge1000,
                        "interventions": interventions,
                        "interventions_lt600": interventions_lt600,
                        "interventions_ge600": interventions_ge600,
                        "branches_evaluated": evaluated,
                        "branches_skipped": skipped,
                        "native_v8_kernel_rows": kernel_rows,
                        "mirror_identity_frames": mirror_frames,
                        "respawns_per_slot": respawns,
                        "timing": {
                            "main_seconds": main_seconds,
                            "search_seconds": search_seconds,
                            "search_seconds_lt600": search_seconds_lt600,
                            "search_seconds_600_999": search_seconds_600_999,
                            "search_seconds_ge1000": search_seconds_ge1000,
                            "branches_lt600": branches_lt600,
                            "branches_600_999": branches_600_999,
                            "branches_ge1000": branches_ge1000,
                        },
                    }
                    completed_arms[arm] = arm_summary
                    rosters.append(roster)
                finally:
                    session.close()

                # Arm boundary checkpoint & signal/stop checks
                boundary_ckpt = _write_ckpt(
                    current_arm=None,
                    current_frame=horizon,
                    in_progress_arm=None,
                    run_started_mono=run_started_mono,
                )
                _poll_external_signal()
                if signal_state["signum"] is not None:
                    sig_num = int(signal_state["signum"])
                    raise CheckpointInterrupted(
                        f"signal {sig_num} received at end of arm {arm}",
                        checkpoint_path=boundary_ckpt or Path(""),
                        arm=arm,
                        frame_index=horizon,
                        signal_number=sig_num,
                        reason=f"signal:{sig_num}",
                        ignored_checkpoints=ignored_checkpoints,
                    )
                if stop_predicate is not None and stop_predicate(arm, horizon):
                    raise CheckpointInterrupted(
                        f"stopped at requested arm boundary ({arm}, {horizon})",
                        checkpoint_path=boundary_ckpt or Path(""),
                        arm=arm,
                        frame_index=horizon,
                        reason="stop_at_boundary",
                        ignored_checkpoints=ignored_checkpoints,
                    )

            if restarted_due_to_roster:
                continue

            if not verified_fresh_roster_once and rosters:
                session = open_watch({"seed": int(seed)}, 24, "bare")
                try:
                    fresh_roster = _compute_session_roster(session)
                finally:
                    session.close()
                if any(r != fresh_roster for r in rosters) and loaded_ckpt_path is not None:
                    _record_and_ignore_mismatched_checkpoint(
                        loaded_ckpt_path,
                        loaded_ckpt_data,
                        "checkpoint completed-arm roster does not match newly loaded model roster",
                    )
                    loaded_ckpt_path = None
                    loaded_ckpt_data = None
                    verified_fresh_roster_once = True
                    continue
                verified_fresh_roster_once = True

            break

        if any(r != rosters[0] for r in rosters[1:]):
            raise ScientificFault("arm roster identity mismatch within paired world")

        total_elapsed = elapsed_seconds_accum + (time.monotonic() - run_started_mono)
        world_out: dict[str, Any] = {
            "schema": WORLD_UNIT_SCHEMA,
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "seed_kind": "fixture" if fixture else "primary",
            "index": int(index),
            "seed": int(seed),
            "horizon": int(horizon),
            "fixture": bool(fixture),
            "record_mode": None,
            "arm_order": list(order),
            "roster": rosters[0],
            "arms": {arm: completed_arms[arm] for arm in p.ARMS if arm in completed_arms},
            "elapsed_seconds": total_elapsed,
        }
        if ignored_checkpoints:
            world_out["resume_refused"] = True
            world_out["ignored_checkpoints"] = list(ignored_checkpoints)
        return world_out
    finally:
        if use_watchdog:
            faulthandler.cancel_dump_traceback_later()
        if can_install_signals:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def inject_live_and_mirror_serpentines(session: Any, lengths: Sequence[int]) -> None:
    """Inject serpentine bodies into both GridBatchSim and live GameState and sync carry/veto."""
    import random

    from research.bl_ppo.collector import _prepare_sim

    sim = session.world.sim
    inject_serpentines([sim], tuple(int(x) for x in lengths), box_height=11)
    _prepare_sim(sim)
    ss = int(sim.s)
    game = session.live.game
    game.frame = int(sim.frame[0])
    game.food_manager.food = [(int(c[0]) * ss, int(c[1]) * ss) for c in sim.food_cells[0]]
    game.food_manager._corpse_positions = {
        (int(c[0]) * ss, int(c[1]) * ss) for c in sim.corpse_cells[0]
    }
    for sidx, snake in enumerate(game.snakes):
        n = int(sim.seg_count[0, sidx])
        hp = int(sim.head_ptr[0, sidx])
        rings = (hp - np.arange(n, dtype=np.int64)) % sim.cap
        cells = sim.bodies[0, sidx, rings]
        snake.segments = [(int(c[0]) * ss, int(c[1]) * ss) for c in cells]
        dvec = CARDINAL[int(sim.direction[0, sidx])]
        snake.direction = (int(dvec[0]), int(dvec[1]))
        snake.length = int(sim.length[0, sidx])
        snake.is_alive = bool(sim.alive[0, sidx])
        snake.boost_frames = int(sim.boost_frames[0, sidx])
        snake.frames_since_food = int(sim.frames_since_food[0, sidx])
        snake.respawn_timer = int(sim.respawn_timer[0, sidx])
        snake._reward_prev_length = int(sim._reward_prev_length[0, sidx])
        snake._prev_nearest_enemy_dist = float("inf")
        snake._prev_nearest_enemy_id = None
        snake._carried_selection = None
        if hasattr(snake, "_free_space_cache"):
            delattr(snake, "_free_space_cache")
    rt = session.world.runtime
    rt._carry_valid[...] = False
    rt._alive_post[...] = sim.alive
    rt.memory.prev_dist[...] = float("inf")
    rt.memory.prev_id[...] = -1
    random.setstate(sim._rngs[0]._rng.getstate())
    verify_session_parity(session)


def measure_checkpoint_scaling(
    *,
    seed: int = p.FIXTURE_SEEDS[0],
    hero_lengths: Sequence[int] = (1, 150, 450, 900, 1200, 1500),
    opponent_lengths: Sequence[int] = (900, 700, 500, 400, 300),
    steps_after_inject: int = 5,
    repeats: int = 5,
) -> list[dict[str, Any]]:
    """Measure checkpoint size and save/load timings across snake body lengths on the Mac."""
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="world-ckpt-bench-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for hero_len in hero_lengths:
            lengths = [int(hero_len)] + [int(x) for x in opponent_lengths]
            session = open_watch({"seed": int(seed)}, 24, "bare")
            try:
                if hero_len > 1:
                    inject_live_and_mirror_serpentines(session, lengths)
                for _ in range(int(steps_after_inject)):
                    session.step()

                actual_lengths = [int(s.length) for s in session.live.game.snakes]
                total_segments = sum(len(s.segments) for s in session.live.game.snakes)
                trigger = DenseTrigger(min_length=0, interval=5)
                trigger.first_decision = 0
                counters = _pack_arm_counters(
                    arm="l1",
                    seed=seed,
                    horizon=p.HORIZON,
                    frame_index=int(steps_after_inject),
                    min_length=0,
                    interval=5,
                    trigger=trigger,
                    was_alive=True,
                    previous_alive=[True] * 6,
                    respawns=[0] * 6,
                    roster=_compute_session_roster(session),
                    decisions=int(steps_after_inject),
                    boost=0,
                    deaths=0,
                    alive_frames=int(steps_after_inject),
                    mass_sum=int(actual_lengths[0] * steps_after_inject),
                    batches=1,
                    batches_lt600=int(actual_lengths[0] < 600),
                    batches_ge600=int(actual_lengths[0] >= 600),
                    batches_ge1000=int(actual_lengths[0] >= 1000),
                    interventions=0,
                    interventions_lt600=0,
                    interventions_ge600=0,
                    evaluated=6,
                    branches_lt600=6 if actual_lengths[0] < 600 else 0,
                    branches_600_999=6 if 600 <= actual_lengths[0] < 1000 else 0,
                    branches_ge1000=6 if actual_lengths[0] >= 1000 else 0,
                    skipped=0,
                    kernel_rows=1800,
                    mirror_frames=int(steps_after_inject),
                    main_seconds=0.1,
                    search_seconds=0.2,
                    search_seconds_lt600=0.0,
                    search_seconds_600_999=0.0,
                    search_seconds_ge1000=0.0,
                    action_counts=[steps_after_inject, 0, 0, 0, 0, 0],
                    recent_actions=[0] * int(steps_after_inject),
                    all_frame_masses=[actual_lengths[0]] * int(steps_after_inject),
                    alive_lengths=[actual_lengths[0]] * int(steps_after_inject),
                    death_lengths=[],
                )

                save_ms_samples: list[float] = []
                load_ms_samples: list[float] = []
                size_bytes = 0
                for rep in range(int(repeats)):
                    ckpt_path = tmp_root / f"bench-L{hero_len}-rep{rep}.ckpt"
                    t0 = time.perf_counter()
                    in_prog = {
                        "arm": "l1",
                        "frame_index": int(steps_after_inject),
                        "counters": counters,
                        "session": snapshot_run_session_state(session),
                    }
                    payload = _build_world_checkpoint_payload(
                        checkpoint_seq=rep + 1,
                        index=-1,
                        seed=seed,
                        horizon=p.HORIZON,
                        arm_order=p.ARMS,
                        fixture=True,
                        completed_arms={},
                        rosters=[],
                        elapsed_seconds_accum=0.3,
                        current_arm="l1",
                        current_frame=int(steps_after_inject),
                        in_progress_arm=in_prog,
                    )
                    meta = save_checkpoint(payload, ckpt_path)
                    save_ms_samples.append((time.perf_counter() - t0) * 1000.0)
                    size_bytes = int(meta["size_bytes"])

                    t1 = time.perf_counter()
                    loaded = load_checkpoint(ckpt_path)
                    restore_run_session_state(session, loaded["in_progress_arm"]["session"])
                    load_ms_samples.append((time.perf_counter() - t1) * 1000.0)

                results.append(
                    {
                        "requested_hero_length": int(hero_len),
                        "actual_lengths": actual_lengths,
                        "total_segments": int(total_segments),
                        "size_bytes": int(size_bytes),
                        "size_kb": round(size_bytes / 1024.0, 2),
                        "save_ms_median": round(float(np.median(save_ms_samples)), 3),
                        "save_ms_max": round(float(np.max(save_ms_samples)), 3),
                        "load_ms_median": round(float(np.median(load_ms_samples)), 3),
                        "load_ms_max": round(float(np.max(load_ms_samples)), 3),
                    }
                )
            finally:
                session.close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resumable Watch6 paired-world runner with atomic checkpointing."
    )
    parser.add_argument("--seed", type=int, default=p.FIXTURE_SEEDS[0])
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--horizon", type=int, default=p.HORIZON)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--arm-order", default="bare,l2_prime,l1")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input-checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--saved-dir", default=str(DEFAULT_LOCAL_SAVED_DIR))
    args = parser.parse_args(argv)

    configure_cpu()
    if "SNAKE_CKPT_DIR" not in os.environ and Path(args.saved_dir).is_dir():
        os.environ["SNAKE_CKPT_DIR"] = str(args.saved_dir)
    os.environ.setdefault("SNAKE_WATCH_SOURCE", "live")
    arm_order = [a.strip() for a in str(args.arm_order).split(",") if a.strip()]
    resume_dirs = [Path(args.input_checkpoint_dir)] if args.input_checkpoint_dir else []

    try:
        result = run_paired_world_resumable(
            index=int(args.index),
            seed=int(args.seed),
            arm_order=arm_order,
            horizon=int(args.horizon),
            fixture=bool(args.fixture),
            checkpoint_dir=Path(args.checkpoint_dir),
            resume_checkpoint_dirs=resume_dirs,
            checkpoint_interval=int(args.checkpoint_interval),
        )
    except CheckpointInterrupted as exc:
        print(
            json.dumps(
                {
                    "status": "checkpoint_interrupted",
                    "checkpoint_path": str(exc.checkpoint_path),
                    "arm": exc.arm,
                    "frame_index": exc.frame_index,
                    "signal": exc.signal_number,
                    "reason": exc.reason,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 75

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    from tools.ops.package_spec import publish_immutable

    publish_immutable(out_path, raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
