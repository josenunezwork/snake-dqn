"""Persistent prewarmed W=1, K<=1 CPU native/exact-stop H300 contention worker.

This module implements Section B (Row 2: "W1/K1 contention-only arm") of the
2026-10-27 ruling:
- One persistent, prewarmed 1-thread CPU native/exact-stop H300 worker process
  co-resident with the isolated 24 Hz Watch6 workload.
- Existing ruler fixtures are fed as outcome-independent stress inputs in the
  M1 interleaved rotation.
- Replies are polled, validated, and logged on the tick, but NO actions are
  ever applied to the live game (`actions_applied_count == 0`).
- Bounded state-copy, request publication (`K <= 1` outstanding request), and
  non-blocking reply polling run on the tick and are timed separately (wall and
  thread CPU ns) as well as included in the full-tick envelope.
"""

from __future__ import annotations

import multiprocessing
import os
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from research.bl_mac_latency import l1
from research.bl_served_search import native_ruler, ruler
from research.bl_served_search.served_tick_probe import _distribution

DEFAULT_FIXTURES_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/"
    "ruler-fixtures-08-final/ruler-fixtures.json"
)
DEFAULT_REFERENCE_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/"
    "y1-native-mac/ruler-receipt.json"
)
Y1_FIXTURES_SHA256 = "bbe6975642512091ba279d9ff7c870000d68faa18e30063807f47af1352e8574"
Y1_REFERENCE_SHA256 = "b97f73d2384a95455beb3c0d243caba513f5f307edc85d0a8da4b81f5e1ac791"
WARM_S_P95_LIMIT_NS = 250_000_000  # 250 ms warm worker S p95 bar (RULING.md Section B)
DEFAULT_REQUEST_TIMEOUT_NS = 10_000_000_000  # 10 s per H300 request
DEFAULT_STARTUP_TIMEOUT_S = 120.0
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class _CooperativeStop(RuntimeError):
    """Raised inside the worker between rollout frames when shutdown is requested."""


def current_rss_bytes(pid: int | None = None) -> int:
    """Return current resident set size in bytes for ``pid`` (or current process)."""
    try:
        target = os.getpid() if pid is None else int(pid)
        return int(psutil.Process(target).memory_info().rss)
    except (psutil.Error, OSError):
        return 0


def peak_rss_bytes() -> int:
    """Return Darwin peak resident set size (`ru_maxrss`) in bytes for this process."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def load_ruler_stress_inputs(
    fixtures_path: Path | str = DEFAULT_FIXTURES_PATH,
    reference_path: Path | str = DEFAULT_REFERENCE_PATH,
) -> dict[str, Any]:
    """Load and verify the hash-pinned 9-root Y1 fixture bank and Mac action oracle."""
    resolved_fixtures = Path(fixtures_path)
    resolved_reference = Path(reference_path)
    bank, bank_sha = l1._read_completed_json(resolved_fixtures)
    fixtures = ruler.validate_fixtures(bank)
    reference, ref_sha = l1._read_completed_json(resolved_reference)
    if bank_sha != Y1_FIXTURES_SHA256 or ref_sha != Y1_REFERENCE_SHA256:
        raise ValueError("W1/K1 contention arm requires the hash-pinned Y1 ruler fixtures/oracle")
    for mode in ("all", "exact_stop"):
        records = reference["cells"][mode]["records"]
        if {row["fixture_id"] for row in records} != {f["fixture_id"] for f in fixtures}:
            raise ValueError("Y1 oracle fixture coverage mismatch")
        for record in records:
            fixture = next(f for f in fixtures if f["fixture_id"] == record["fixture_id"])
            for batch in record["batches"]:
                native_ruler.validate_batch(batch, fixture, mode)
    if reference["fixtures"] != bank or reference["platform"]["label"] != "mac":
        raise ValueError("W1/K1 requires the nine Y1 fixtures and Mac all-branch oracle")
    actions = {
        str(row["fixture_id"]): int(row["batches"][0]["semantic"]["chosen"])
        for row in reference["cells"]["all"]["records"]
    }
    for row in reference["cells"]["all"]["records"]:
        fixture_id = str(row["fixture_id"])
        if any(int(batch["semantic"]["chosen"]) != actions[fixture_id] for batch in row["batches"]):
            raise ValueError("Y1 all-branch reference actions drift")
    return {
        "fixtures": fixtures,
        "fixtures_path": str(resolved_fixtures.resolve()),
        "fixtures_sha256": bank_sha,
        "reference_path": str(resolved_reference.resolve()),
        "reference_sha256": ref_sha,
        "actions": actions,
    }


def fixture_at(
    fixtures: Sequence[Mapping[str, Any]], request_index: int
) -> tuple[int, dict[str, Any]]:
    """Return ``(repeat, fixture)`` using M1's interleaved root rotation."""
    if not fixtures:
        raise ValueError("fixtures must be non-empty")
    if request_index < 0:
        raise ValueError("request_index must be non-negative")
    count = len(fixtures)
    repeat = request_index // count
    slot = request_index % count
    return repeat, dict(fixtures[(slot + repeat) % count])


def copy_live_game_state(session: Any) -> tuple[dict[str, Any], int]:
    """Copy a bounded, detached snapshot of live Watch6 state without mutating game/RNG."""
    if session is None or getattr(session, "game", None) is None:
        return {"frame": -1, "hero_id": -1, "alive_snakes": 0, "snakes": [], "food": b""}, 0
    game = session.game
    hero_id = int(getattr(session, "hero_id", 0))
    frame = int(getattr(game, "frame", 0))
    alive_snakes = int(getattr(game, "alive_snakes", 0))
    snakes_payload: list[dict[str, Any]] = []
    total_bytes = 24
    for snake in tuple(getattr(game, "snakes", ())):
        raw_segments = getattr(snake, "segments", None)
        if raw_segments is None:
            raw_segments = getattr(snake, "body", ())
        if raw_segments:
            seg_bytes = np.asarray(raw_segments, dtype=np.float32).tobytes()
        else:
            seg_bytes = b""
        direction = getattr(snake, "direction", (0, 0))
        if isinstance(direction, (list, tuple)) and len(direction) >= 2:
            dir_tuple = (float(direction[0]), float(direction[1]))
        else:
            dir_tuple = (0.0, 0.0)
        last_action = getattr(snake, "last_action", None)
        snakes_payload.append(
            {
                "id": int(getattr(snake, "id", -1)),
                "length": int(getattr(snake, "length", 0)),
                "is_alive": bool(getattr(snake, "is_alive", False)),
                "is_boosting": bool(getattr(snake, "is_boosting", False)),
                "direction": dir_tuple,
                "last_action": None if last_action is None else int(last_action),
                "segments_f32": seg_bytes,
            }
        )
        total_bytes += len(seg_bytes) + 48
    raw_food = getattr(game, "food", ())
    if raw_food:
        coords = [
            (
                float(getattr(item, "x", item[0] if isinstance(item, (list, tuple)) else 0.0)),
                float(getattr(item, "y", item[1] if isinstance(item, (list, tuple)) else 0.0)),
            )
            for item in raw_food
        ]
        food_bytes = np.asarray(coords, dtype=np.float32).tobytes()
    else:
        food_bytes = b""
    total_bytes += len(food_bytes)
    snapshot = {
        "frame": frame,
        "hero_id": hero_id,
        "alive_snakes": alive_snakes,
        "snakes": snakes_payload,
        "food_f32": food_bytes,
    }
    return snapshot, total_bytes


class _SyntheticFastOperator:
    """Fast deterministic operator for unit tests of IPC/queue/lifecycle mechanics."""

    def __init__(self, delay_seconds: float = 0.0, chosen_override: int | None = None) -> None:
        self.delay_seconds = float(delay_seconds)
        self.chosen_override = chosen_override

    def batch(self, fixture: dict[str, Any]) -> dict[str, Any]:
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        legal = [int(a) for a in fixture.get("legal_actions", [0, 1, 2])]
        bare = int(legal[0])
        chosen = (
            int(self.chosen_override)
            if self.chosen_override is not None and int(self.chosen_override) in legal
            else int(fixture.get("expected_action", bare))
        )
        order = [bare] if chosen == bare else [bare, chosen]
        outcomes: dict[str, Any] = {}
        total_frames = 0
        for action in sorted(legal):
            if action in order:
                survives = action == chosen
                frames = 300 if survives else 120
                outcomes[str(action)] = {"survives": survives, "frames_executed": frames}
                total_frames += frames
            else:
                outcomes[str(action)] = {"status": "NOT_EVALUATED"}
        is_scripted = fixture.get("mix") == "scripted"
        return {
            "semantic": {
                "fixture_id": str(fixture["fixture_id"]),
                "bare": bare,
                "chosen": chosen,
                "fired": True,
                "search_mode": "exact_stop",
                "outcomes": outcomes,
                "evaluation_order": order,
                "branches_evaluated": len(order),
                "branches_skipped": len(legal) - len(order),
                "rollout_branch_frames": total_frames,
            },
            "stage_seconds": {
                "sim": 0.001,
                "observation": 0.0002,
                "hero_v8": 0.0001,
                "opponent_v8": 0.0001,
                "vector61_frp3": 0.0002,
                "b8_forward": 0.0003,
                "clone_digest": 0.0001,
                "submit_visible": 0.0,
                "submit_consumed": 0.0,
            },
            "native_v8": {
                "mode": "kernel",
                "calls": 0 if is_scripted else 300,
                "kernel_rows": 0 if is_scripted else 1200,
                "stock_rows": 0,
                "noncandidate_rows": 0,
                "fallback_rows": 0,
            },
        }


def _worker_main(
    conn: Any,
    stop_event: Any,
    operator_mode: str = "exact_stop",
    operator_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Run the persistent single-thread CPU native/exact-stop worker loop."""
    kwargs = dict(operator_kwargs or {})
    try:
        for name in THREAD_ENV:
            os.environ[name] = "1"
        os.environ["SNAKE_DQN_DEVICE"] = "cpu"
        if "RULER_B8_PATH" not in os.environ:
            default_b8 = Path("saved_snakes/b8_s3_t7_20261016.pth")
            if default_b8.is_file():
                os.environ["RULER_B8_PATH"] = str(default_b8.resolve())
        from research.t1_commitment_20261015.replay import configure_cpu

        configure_cpu()
        import numba
        import torch

        numba.set_num_threads(1)
        if operator_mode == "exact_stop":
            from research.bl_served_search.ruler_operator import create_stop_operator

            operator: Any = create_stop_operator()
        elif operator_mode == "synthetic_fast":
            operator = _SyntheticFastOperator(**kwargs)
        else:
            raise ValueError(f"unknown worker operator_mode: {operator_mode!r}")

        conn.send(
            {
                "kind": "ready",
                "ok": True,
                "worker_pid": os.getpid(),
                "rss_bytes": current_rss_bytes(),
                "maxrss_bytes": peak_rss_bytes(),
                "threads": {
                    "environment": {name: os.environ.get(name) for name in THREAD_ENV},
                    "device": os.environ.get("SNAKE_DQN_DEVICE"),
                    "torch_num_threads": int(torch.get_num_threads()),
                    "torch_num_interop_threads": int(torch.get_num_interop_threads()),
                    "numba_num_threads": int(numba.get_num_threads()),
                },
            }
        )

        from research.t3_rollout_labels_20261015 import bank_engine

        originals = {
            name: getattr(bank_engine, name) for name in ("choose_and_step", "step_prepared")
        }

        def guarded(original: Callable[..., Any]) -> Callable[..., Any]:
            def step(*args: Any, **step_kwargs: Any) -> Any:
                if stop_event.is_set():
                    raise _CooperativeStop("worker stopped before next rollout step")
                return original(*args, **step_kwargs)

            return step

        while not stop_event.is_set():
            if not conn.poll(0.02):
                continue
            try:
                message = conn.recv()
            except EOFError:
                break
            if not isinstance(message, Mapping):
                continue
            cmd = message.get("cmd")
            if cmd == "shutdown":
                conn.send(
                    {
                        "kind": "shutdown_ack",
                        "ok": True,
                        "worker_pid": os.getpid(),
                        "rss_bytes": current_rss_bytes(),
                        "maxrss_bytes": peak_rss_bytes(),
                    }
                )
                break
            if cmd == "simulate_crash":
                os._exit(int(message.get("exitcode", 137)))
            if cmd == "simulate_hang":
                hang_seconds = float(message.get("seconds", 30.0))
                stop_event.wait(hang_seconds)
                continue
            if cmd == "simulate_invalid_reply":
                conn.send(
                    {
                        "kind": "reply",
                        "ok": True,
                        "request_id": int(message.get("request_id", -1)),
                        "malformed": True,
                    }
                )
                continue
            if cmd == "evaluate":
                fixture = dict(message["fixture"])
                state_snapshot = message.get("state_snapshot")
                if not isinstance(state_snapshot, Mapping):
                    conn.send(
                        {
                            "kind": "reply",
                            "ok": False,
                            "request_id": int(message.get("request_id", -1)),
                            "error": "missing state_snapshot in evaluate packet",
                        }
                    )
                    continue
                started_wall = time.perf_counter_ns()
                started_cpu = time.thread_time_ns()
                try:
                    for name, original in originals.items():
                        setattr(bank_engine, name, guarded(original))
                    batch = operator.batch(fixture)
                except _CooperativeStop:
                    conn.send(
                        {
                            "kind": "stopped",
                            "ok": True,
                            "request_id": int(message.get("request_id", -1)),
                            "worker_pid": os.getpid(),
                            "rss_bytes": current_rss_bytes(),
                            "maxrss_bytes": peak_rss_bytes(),
                        }
                    )
                    break
                finally:
                    for name, original in originals.items():
                        setattr(bank_engine, name, original)
                completed_wall = time.perf_counter_ns()
                completed_cpu = time.thread_time_ns()
                service_ns = max(1, completed_wall - started_wall)
                cpu_ns = max(0, completed_cpu - started_cpu)
                conn.send(
                    {
                        "kind": "reply",
                        "ok": True,
                        "request_id": int(message["request_id"]),
                        "tick_index": int(message["tick_index"]),
                        "repeat": int(message["repeat"]),
                        "phase": str(message["phase"]),
                        "fixture_id": str(fixture["fixture_id"]),
                        "admitted_ns": int(message["admitted_ns"]),
                        "state_copy_bytes": int(message.get("state_copy_bytes", 0)),
                        "state_frame": int(state_snapshot.get("frame", -1)),
                        "worker_pid": os.getpid(),
                        "worker_started_ns": started_wall,
                        "worker_completed_ns": completed_wall,
                        "worker_service_ns": service_ns,
                        "worker_thread_cpu_ns": cpu_ns,
                        "worker_wall_seconds": service_ns / 1e9,
                        "worker_threads": {
                            "environment": {name: os.environ.get(name) for name in THREAD_ENV},
                            "torch": int(torch.get_num_threads()),
                            "numba": int(numba.get_num_threads()),
                        },
                        "worker_rss_bytes": current_rss_bytes(),
                        "worker_maxrss_bytes": peak_rss_bytes(),
                        "semantic": batch["semantic"],
                        "native_v8": batch["native_v8"],
                        "stage_seconds": batch["stage_seconds"],
                    }
                )
    except BaseException as exc:
        try:
            conn.send(
                {"kind": "worker_error", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
        except BaseException:
            pass
    finally:
        try:
            conn.close()
        except BaseException:
            pass


class PersistentExactStopWorker:
    """Parent-side coordinator for the persistent W=1, K<=1 contention-only worker."""

    def __init__(
        self,
        *,
        fixtures: Sequence[Mapping[str, Any]],
        reference_actions: Mapping[str, int],
        measured_ticks: int,
        fixtures_sha256: str = Y1_FIXTURES_SHA256,
        reference_sha256: str = Y1_REFERENCE_SHA256,
        max_pending: int = 1,
        worker_mode: str = "active",
        request_timeout_ns: int = DEFAULT_REQUEST_TIMEOUT_NS,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        operator_mode: str = "exact_stop",
        operator_kwargs: Mapping[str, Any] | None = None,
        validate_batches: bool = True,
    ) -> None:
        if max_pending != 1:
            raise ValueError("W1/K1 contention arm requires max_pending == 1 (K <= 1)")
        if worker_mode not in ("active", "idle"):
            raise ValueError(f"unsupported worker_mode: {worker_mode!r}")
        if measured_ticks < 0:
            raise ValueError("measured_ticks must be non-negative")
        if not fixtures:
            raise ValueError("fixtures must be non-empty")
        self.fixtures = [dict(f) for f in fixtures]
        self._fixtures_by_id = {str(f["fixture_id"]): dict(f) for f in self.fixtures}
        self.reference_actions = {str(k): int(v) for k, v in reference_actions.items()}
        self.fixtures_sha256 = str(fixtures_sha256)
        self.reference_sha256 = str(reference_sha256)
        self.measured_ticks = int(measured_ticks)
        self.max_pending = 1
        self.worker_mode = worker_mode
        self.request_timeout_ns = int(request_timeout_ns)
        self.startup_timeout_s = float(startup_timeout_s)
        self.operator_mode = operator_mode
        self.operator_kwargs = dict(operator_kwargs or {})
        self.validate_batches = bool(validate_batches)

        self._ctx = multiprocessing.get_context("spawn")
        self._parent_conn: Any | None = None
        self._stop_event: Any = self._ctx.Event()
        self._process: Any | None = None

        self.worker_pid: int | None = None
        self.worker_threads: dict[str, Any] = {}
        self.worker_startup_wall_ns: int = 0
        self.prewarm_wall_ns: int = 0
        self.prewarmed: bool = False

        self.worker_rss_bytes_initial: int = 0
        self.worker_rss_bytes_after_prewarm: int = 0
        self.worker_rss_bytes_final: int = 0
        self.worker_maxrss_bytes: int = 0

        self.pending_request: dict[str, Any] | None = None
        self.current_pending_count: int = 0
        self.max_observed_pending: int = 0
        self._next_request_id: int = 0
        self._schedule_cursor: int = 0

        self.prewarm_completed_count: int = 0
        self.offered_count: int = 0
        self.submitted_count: int = 0
        self.completed_count: int = 0
        self.in_flight_at_end_count: int = 0
        self.queue_overflow_count: int = 0
        self.worker_timeout_count: int = 0
        self.worker_crash_count: int = 0
        self.invalid_reply_count: int = 0
        self.reference_mismatch_count: int = 0
        self.fallback_count: int = 0
        self.actions_applied_count: int = 0

        self._worker_dead: bool = False
        self._worker_timed_out: bool = False
        self.clean_shutdown: bool = False
        self.worker_exitcode: int | None = None

        self.state_copy_wall_ns: list[int] = [0] * self.measured_ticks
        self.state_copy_thread_cpu_ns: list[int] = [0] * self.measured_ticks
        self.state_copy_bytes: list[int] = [0] * self.measured_ticks
        self.publish_wall_ns: list[int] = [0] * self.measured_ticks
        self.publish_thread_cpu_ns: list[int] = [0] * self.measured_ticks
        self.poll_wall_ns: list[int] = [0] * self.measured_ticks
        self.poll_thread_cpu_ns: list[int] = [0] * self.measured_ticks
        self.tick_admitted: list[int] = [0] * self.measured_ticks
        self.tick_completed: list[int] = [0] * self.measured_ticks
        self.tick_overflow: list[int] = [0] * self.measured_ticks

        self.cold_records: list[dict[str, Any]] = []
        self.warm_records: list[dict[str, Any]] = []

    def start_and_prewarm(self, prewarm_count: int | None = None) -> dict[str, Any]:
        """Spawn the persistent worker, attest 1-thread CPU config, and prewarm cold roots."""
        if self._process is not None:
            raise RuntimeError("worker already started")
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        self._parent_conn = parent_conn
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, self._stop_event, self.operator_mode, self.operator_kwargs),
            name="served-tick-w1-k1-worker",
            daemon=False,
        )
        startup_started = time.perf_counter_ns()
        self._process.start()
        child_conn.close()

        if not parent_conn.poll(self.startup_timeout_s):
            self.close()
            raise TimeoutError(
                f"W1/K1 worker failed to complete startup within {self.startup_timeout_s}s"
            )
        ready = parent_conn.recv()
        if not isinstance(ready, Mapping) or ready.get("kind") != "ready" or not ready.get("ok"):
            self.close()
            raise RuntimeError(f"W1/K1 worker startup failed: {ready}")
        self.worker_startup_wall_ns = time.perf_counter_ns() - startup_started
        self.worker_pid = int(ready["worker_pid"])
        self.worker_threads = dict(ready.get("threads", {}))
        self.worker_rss_bytes_initial = int(ready.get("rss_bytes", 0))
        self.worker_maxrss_bytes = max(
            self.worker_maxrss_bytes,
            int(ready.get("maxrss_bytes", 0)),
            self.worker_rss_bytes_initial,
        )
        env_threads = self.worker_threads.get("environment", {})
        if (
            self.worker_threads.get("device") != "cpu"
            or any(env_threads.get(k) != "1" for k in THREAD_ENV)
            or self.worker_threads.get("torch_num_threads") != 1
            or self.worker_threads.get("torch_num_interop_threads") != 1
            or self.worker_threads.get("numba_num_threads") != 1
        ):
            self.close()
            raise RuntimeError(
                f"W1/K1 worker failed 1-thread CPU attestation: {self.worker_threads}"
            )

        target_prewarm = len(self.fixtures) if prewarm_count is None else int(prewarm_count)
        prewarm_started = time.perf_counter_ns()
        for index in range(target_prewarm):
            fixture = self.fixtures[index % len(self.fixtures)]
            request_id = self._next_request_id
            self._next_request_id += 1
            admitted_ns = time.perf_counter_ns()
            packet = {
                "cmd": "evaluate",
                "request_id": request_id,
                "tick_index": -1 - index,
                "repeat": 0,
                "phase": "cold",
                "admitted_ns": admitted_ns,
                "fixture": fixture,
                "state_snapshot": {
                    "frame": -1,
                    "hero_id": 0,
                    "alive_snakes": 6,
                    "snakes": [],
                    "food_f32": b"",
                },
                "state_copy_bytes": 0,
            }
            parent_conn.send(packet)
            if not parent_conn.poll(self.startup_timeout_s):
                self.close()
                raise TimeoutError(
                    f"W1/K1 worker prewarm timed out on fixture {fixture['fixture_id']}"
                )
            reply = parent_conn.recv()
            visible_ns = time.perf_counter_ns()
            record = self._validate_and_record_reply(
                reply, visible_ns=visible_ns, expected_request_id=request_id, expected_phase="cold"
            )
            if record is None:
                self.close()
                raise RuntimeError(f"W1/K1 worker prewarm returned invalid reply: {reply}")
            self.prewarm_completed_count += 1
        self.prewarm_wall_ns = time.perf_counter_ns() - prewarm_started
        self.worker_rss_bytes_after_prewarm = (
            int(self.cold_records[-1]["worker_rss_bytes"])
            if self.cold_records
            else current_rss_bytes(self.worker_pid)
        )
        self.prewarmed = True
        self.pending_request = None
        self.current_pending_count = 0
        return {
            "worker_pid": self.worker_pid,
            "worker_startup_wall_ns": self.worker_startup_wall_ns,
            "prewarm_wall_ns": self.prewarm_wall_ns,
            "prewarm_completed_count": self.prewarm_completed_count,
            "worker_rss_bytes_initial": self.worker_rss_bytes_initial,
            "worker_rss_bytes_after_prewarm": self.worker_rss_bytes_after_prewarm,
        }

    def _validate_and_record_reply(
        self,
        reply: Any,
        *,
        visible_ns: int,
        expected_request_id: int,
        expected_phase: str | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(reply, Mapping) or reply.get("kind") != "reply" or not reply.get("ok"):
            self.invalid_reply_count += 1
            self.fallback_count += 1
            return None
        if int(reply.get("request_id", -1)) != int(expected_request_id):
            self.invalid_reply_count += 1
            self.fallback_count += 1
            return None
        fixture_id = str(reply.get("fixture_id", ""))
        fixture = self._fixtures_by_id.get(fixture_id)
        if fixture is None:
            self.invalid_reply_count += 1
            self.fallback_count += 1
            return None
        if self.validate_batches:
            try:
                native_ruler.validate_batch(dict(reply), fixture, "exact_stop")
            except (KeyError, TypeError, ValueError):
                self.invalid_reply_count += 1
                self.fallback_count += 1
                return None
        semantic = reply.get("semantic", {})
        if not isinstance(semantic, Mapping) or "chosen" not in semantic or "bare" not in semantic:
            self.invalid_reply_count += 1
            self.fallback_count += 1
            return None
        chosen = int(semantic["chosen"])
        bare = int(semantic["bare"])
        expected_action = self.reference_actions.get(fixture_id)
        reference_match = expected_action is None or chosen == int(expected_action)
        if not reference_match:
            self.reference_mismatch_count += 1
            self.fallback_count += 1
            return None
        admitted_ns = int(reply["admitted_ns"])
        service_ns = int(reply["worker_service_ns"])
        cpu_ns = int(reply.get("worker_thread_cpu_ns", 0))
        s_ns = max(0, int(visible_ns) - admitted_ns)
        rss_bytes = int(reply.get("worker_rss_bytes", 0))
        maxrss_bytes = int(reply.get("worker_maxrss_bytes", 0))
        self.worker_maxrss_bytes = max(self.worker_maxrss_bytes, rss_bytes, maxrss_bytes)
        phase = str(reply.get("phase", expected_phase or "warm"))
        # Contention-only invariant: replies are consumed and logged, never applied.
        record = {
            "request_id": int(reply["request_id"]),
            "tick_index": int(reply["tick_index"]),
            "repeat": int(reply["repeat"]),
            "phase": phase,
            "fixture_id": fixture_id,
            "admitted_ns": admitted_ns,
            "visible_ns": int(visible_ns),
            "S_ns": s_ns,
            "worker_service_ns": service_ns,
            "worker_thread_cpu_ns": cpu_ns,
            "worker_rss_bytes": rss_bytes,
            "state_copy_bytes": int(reply.get("state_copy_bytes", 0)),
            "chosen_action": chosen,
            "bare_action": bare,
            "reference_action": expected_action,
            "reference_match": reference_match,
            "action_applied": False,
            "branches_evaluated": int(semantic.get("branches_evaluated", 0)),
            "branches_skipped": int(semantic.get("branches_skipped", 0)),
            "rollout_branch_frames": int(semantic.get("rollout_branch_frames", 0)),
            "native_v8": dict(reply.get("native_v8", {})),
            "stage_seconds": dict(reply.get("stage_seconds", {})),
        }
        if phase == "cold":
            self.cold_records.append(record)
        else:
            self.warm_records.append(record)
        return record

    def poll(self, tick_index: int) -> dict[str, Any] | None:
        """Non-blockingly poll the worker pipe on the tick; consume/log reply, apply NO action."""
        started_wall = time.perf_counter_ns()
        started_cpu = time.thread_time_ns()
        record: dict[str, Any] | None = None
        try:
            if self._parent_conn is None or self.pending_request is None:
                return None
            try:
                has_message = bool(self._parent_conn.poll(0))
            except (BrokenPipeError, EOFError, OSError):
                self._handle_worker_crash()
                return None
            if has_message:
                try:
                    reply = self._parent_conn.recv()
                except (BrokenPipeError, EOFError, OSError):
                    self._handle_worker_crash()
                    return None
                visible_ns = time.perf_counter_ns()
                expected_id = int(self.pending_request["request_id"])
                self.pending_request = None
                self.current_pending_count = 0
                record = self._validate_and_record_reply(
                    reply, visible_ns=visible_ns, expected_request_id=expected_id
                )
                if record is not None:
                    self.completed_count += 1
                    if 0 <= tick_index < self.measured_ticks:
                        self.tick_completed[tick_index] = 1
                return record
            # No message ready yet: check whether the worker crashed or timed out.
            if self._process is None or not self._process.is_alive():
                self._handle_worker_crash()
                return None
            elapsed_ns = time.perf_counter_ns() - int(self.pending_request["admitted_ns"])
            if elapsed_ns > self.request_timeout_ns:
                self._handle_worker_timeout()
                return None
            return None
        finally:
            if 0 <= tick_index < self.measured_ticks:
                self.poll_wall_ns[tick_index] += max(0, time.perf_counter_ns() - started_wall)
                self.poll_thread_cpu_ns[tick_index] += max(0, time.thread_time_ns() - started_cpu)

    def _handle_worker_crash(self) -> None:
        self.pending_request = None
        self.current_pending_count = 0
        if not self._worker_dead:
            self._worker_dead = True
            self.worker_crash_count += 1
            self.fallback_count += 1

    def _handle_worker_timeout(self) -> None:
        self.pending_request = None
        self.current_pending_count = 0
        if not self._worker_timed_out:
            self._worker_timed_out = True
            self.worker_timeout_count += 1
            self.fallback_count += 1
            self._stop_event.set()

    def copy_state_and_prepare_request(
        self, session: Any, tick_index: int
    ) -> tuple[dict[str, Any], int]:
        """Copy detached live game state and pair with the next scheduled fixture."""
        started_wall = time.perf_counter_ns()
        started_cpu = time.thread_time_ns()
        try:
            state_snapshot, copy_bytes = copy_live_game_state(session)
            repeat, fixture = fixture_at(self.fixtures, self._schedule_cursor)
            phase = "warm" if (self.prewarmed or repeat > 0) else "cold"
            packet = {
                "cmd": "evaluate",
                "tick_index": int(tick_index),
                "repeat": repeat,
                "phase": phase,
                "fixture": fixture,
                "state_snapshot": state_snapshot,
                "state_copy_bytes": copy_bytes,
            }
            return packet, copy_bytes
        finally:
            if 0 <= tick_index < self.measured_ticks:
                self.state_copy_wall_ns[tick_index] += max(0, time.perf_counter_ns() - started_wall)
                self.state_copy_thread_cpu_ns[tick_index] += max(
                    0, time.thread_time_ns() - started_cpu
                )
                self.state_copy_bytes[tick_index] = copy_bytes

    def try_publish(self, packet: Mapping[str, Any], copy_bytes: int, tick_index: int) -> bool:
        """Enforce K <= 1 bounded queue; send over Pipe if idle or drop-while-busy if pending."""
        started_wall = time.perf_counter_ns()
        started_cpu = time.thread_time_ns()
        try:
            self.offered_count += 1
            if self._worker_dead or self._worker_timed_out or self._parent_conn is None:
                self.fallback_count += 1
                return False
            if self._process is None or not self._process.is_alive():
                self._handle_worker_crash()
                return False
            if self.current_pending_count >= self.max_pending:
                # K <= 1 backpressure: refuse immediately without blocking or queuing.
                self.queue_overflow_count += 1
                self.fallback_count += 1
                if 0 <= tick_index < self.measured_ticks:
                    self.tick_overflow[tick_index] = 1
                return False
            request_id = self._next_request_id
            self._next_request_id += 1
            admitted_ns = time.perf_counter_ns()
            outbound = {
                **packet,
                "request_id": request_id,
                "admitted_ns": admitted_ns,
                "state_copy_bytes": int(copy_bytes),
            }
            try:
                self._parent_conn.send(outbound)
            except (BrokenPipeError, EOFError, OSError):
                self._handle_worker_crash()
                return False
            self.pending_request = {
                "request_id": request_id,
                "tick_index": int(tick_index),
                "admitted_ns": admitted_ns,
                "fixture_id": str(packet["fixture"]["fixture_id"]),
            }
            self.current_pending_count = 1
            if self.current_pending_count > self.max_observed_pending:
                self.max_observed_pending = self.current_pending_count
            self.submitted_count += 1
            self._schedule_cursor += 1
            if 0 <= tick_index < self.measured_ticks:
                self.tick_admitted[tick_index] = 1
            return True
        finally:
            if 0 <= tick_index < self.measured_ticks:
                self.publish_wall_ns[tick_index] += max(0, time.perf_counter_ns() - started_wall)
                self.publish_thread_cpu_ns[tick_index] += max(
                    0, time.thread_time_ns() - started_cpu
                )

    def step_tick(self, session: Any, tick_index: int) -> None:
        """Execute bounded poll, state-copy, and K<=1 publish on a measured tick."""
        self.poll(tick_index)
        if self.worker_mode == "active":
            packet, copy_bytes = self.copy_state_and_prepare_request(session, tick_index)
            self.try_publish(packet, copy_bytes, tick_index)

    def close(self) -> dict[str, Any]:
        """Cooperatively stop any in-flight rollout and shut down the persistent worker."""
        self.in_flight_at_end_count = int(self.current_pending_count)
        if self.worker_pid is not None and self._process is not None and self._process.is_alive():
            rss_now = current_rss_bytes(self.worker_pid)
            if rss_now > 0:
                self.worker_rss_bytes_final = rss_now
                self.worker_maxrss_bytes = max(self.worker_maxrss_bytes, rss_now)
        self._stop_event.set()
        if self._parent_conn is not None:
            try:
                if self._process is not None and self._process.is_alive():
                    self._parent_conn.send({"cmd": "shutdown"})
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline and self._parent_conn.poll(0.05):
                        msg = self._parent_conn.recv()
                        if isinstance(msg, Mapping) and msg.get("kind") in (
                            "shutdown_ack",
                            "stopped",
                        ):
                            rss_end = int(msg.get("rss_bytes", 0))
                            maxrss_end = int(msg.get("maxrss_bytes", 0))
                            if rss_end > 0:
                                self.worker_rss_bytes_final = rss_end
                            self.worker_maxrss_bytes = max(
                                self.worker_maxrss_bytes, rss_end, maxrss_end
                            )
                            break
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                try:
                    self._parent_conn.close()
                except BaseException:
                    pass
                self._parent_conn = None
        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
            self.worker_exitcode = self._process.exitcode
            self.clean_shutdown = (
                self.worker_exitcode == 0 and not self._worker_dead and not self._worker_timed_out
            )
        self.pending_request = None
        self.current_pending_count = 0
        if self.worker_rss_bytes_final == 0 and self.warm_records:
            self.worker_rss_bytes_final = int(self.warm_records[-1]["worker_rss_bytes"])
        elif self.worker_rss_bytes_final == 0:
            self.worker_rss_bytes_final = self.worker_rss_bytes_after_prewarm
        return {
            "worker_pid": self.worker_pid,
            "worker_exitcode": self.worker_exitcode,
            "clean_shutdown": self.clean_shutdown,
            "worker_rss_bytes_final": self.worker_rss_bytes_final,
            "worker_maxrss_bytes": self.worker_maxrss_bytes,
        }

    def export_payload(
        self,
        *,
        game_rss_before: int,
        game_rss_after: int,
        game_maxrss: int,
    ) -> dict[str, Any]:
        """Export raw and summarized contention worker telemetry after the cell stops."""
        raw_payload = {
            "schema": "served-tick-w1-k1-worker/v1",
            "worker_mode": self.worker_mode,
            "operator_mode": self.operator_mode,
            "workers": 1,
            "max_pending": self.max_pending,
            "prewarmed": self.prewarmed,
            "worker_pid": self.worker_pid,
            "worker_exitcode": self.worker_exitcode,
            "clean_shutdown": self.clean_shutdown,
            "worker_startup_wall_ns": self.worker_startup_wall_ns,
            "prewarm_wall_ns": self.prewarm_wall_ns,
            "worker_threads": self.worker_threads,
            "fixtures_sha256": self.fixtures_sha256,
            "reference_sha256": self.reference_sha256,
            "fixture_ids": [str(f["fixture_id"]) for f in self.fixtures],
            "counts": {
                "max_pending": self.max_pending,
                "max_observed_pending": self.max_observed_pending,
                "prewarm_completed_count": self.prewarm_completed_count,
                "offered_count": self.offered_count,
                "submitted_count": self.submitted_count,
                "completed_count": self.completed_count,
                "in_flight_at_end_count": self.in_flight_at_end_count,
                "queue_overflow_count": self.queue_overflow_count,
                "worker_timeout_count": self.worker_timeout_count,
                "worker_crash_count": self.worker_crash_count,
                "invalid_reply_count": self.invalid_reply_count,
                "reference_mismatch_count": self.reference_mismatch_count,
                "fallback_count": self.fallback_count,
                "actions_applied_count": self.actions_applied_count,
            },
            "rss": {
                "game_rss_bytes_before": int(game_rss_before),
                "game_rss_bytes_after": int(game_rss_after),
                "game_rss_bytes_delta": int(game_rss_after) - int(game_rss_before),
                "game_maxrss_bytes": int(game_maxrss),
                "worker_rss_bytes_initial": int(self.worker_rss_bytes_initial),
                "worker_rss_bytes_after_prewarm": int(self.worker_rss_bytes_after_prewarm),
                "worker_rss_bytes_final": int(self.worker_rss_bytes_final),
                "worker_maxrss_bytes": int(self.worker_maxrss_bytes),
            },
            "tick_costs": {
                "state_copy_wall_ns": list(self.state_copy_wall_ns),
                "state_copy_thread_cpu_ns": list(self.state_copy_thread_cpu_ns),
                "state_copy_bytes": list(self.state_copy_bytes),
                "publish_wall_ns": list(self.publish_wall_ns),
                "publish_thread_cpu_ns": list(self.publish_thread_cpu_ns),
                "poll_wall_ns": list(self.poll_wall_ns),
                "poll_thread_cpu_ns": list(self.poll_thread_cpu_ns),
                "tick_admitted": list(self.tick_admitted),
                "tick_completed": list(self.tick_completed),
                "tick_overflow": list(self.tick_overflow),
            },
            "cold_records": list(self.cold_records),
            "warm_records": list(self.warm_records),
        }
        raw_payload["summary"] = reduce_worker_contention(
            raw_payload, expected_ticks=self.measured_ticks
        )
        return raw_payload


def reduce_worker_contention(
    worker_data: Mapping[str, Any],
    *,
    expected_ticks: int,
) -> dict[str, Any]:
    """Validate and summarize W1/K1 contention telemetry fail-closed."""
    if (
        not isinstance(worker_data, Mapping)
        or worker_data.get("schema") != "served-tick-w1-k1-worker/v1"
    ):
        return {"verdict": "INVALID", "reason": "missing or invalid W1/K1 worker schema"}
    worker_mode = worker_data.get("worker_mode")
    if worker_mode not in ("active", "idle"):
        return {"verdict": "INVALID", "reason": "invalid W1/K1 worker_mode"}
    if worker_data.get("workers") != 1 or worker_data.get("max_pending") != 1:
        return {"verdict": "INVALID", "reason": "W1/K1 worker/max_pending contract violated"}
    if not worker_data.get("prewarmed") or not worker_data.get("clean_shutdown"):
        return {"verdict": "INVALID", "reason": "W1/K1 worker not prewarmed or unclean shutdown"}
    threads = worker_data.get("worker_threads", {})
    env_threads = threads.get("environment", {}) if isinstance(threads, Mapping) else {}
    if (
        not isinstance(threads, Mapping)
        or threads.get("device") != "cpu"
        or any(env_threads.get(k) != "1" for k in THREAD_ENV)
        or threads.get("torch_num_threads") != 1
        or threads.get("torch_num_interop_threads") != 1
        or threads.get("numba_num_threads") != 1
    ):
        return {"verdict": "INVALID", "reason": "W1/K1 worker thread/device attestation failed"}

    counts = worker_data.get("counts", {})
    if not isinstance(counts, Mapping):
        return {"verdict": "INVALID", "reason": "missing W1/K1 counts"}
    required_counts = (
        "max_pending",
        "max_observed_pending",
        "prewarm_completed_count",
        "offered_count",
        "submitted_count",
        "completed_count",
        "in_flight_at_end_count",
        "queue_overflow_count",
        "worker_timeout_count",
        "worker_crash_count",
        "invalid_reply_count",
        "reference_mismatch_count",
        "fallback_count",
        "actions_applied_count",
    )
    if any(type(counts.get(k)) is not int or counts[k] < 0 for k in required_counts):
        return {"verdict": "INVALID", "reason": "malformed W1/K1 integer counters"}
    if counts["max_pending"] != 1 or counts["max_observed_pending"] > 1:
        return {"verdict": "INVALID", "reason": "K <= 1 bounded queue violated"}
    if counts["actions_applied_count"] != 0:
        return {"verdict": "INVALID", "reason": "contention-only arm applied a worker action"}
    if (
        counts["worker_timeout_count"] > 0
        or counts["worker_crash_count"] > 0
        or counts["invalid_reply_count"] > 0
        or counts["reference_mismatch_count"] > 0
    ):
        return {
            "verdict": "INVALID",
            "reason": "worker crash, timeout, or invalid/mismatched reply",
        }

    tick_costs = worker_data.get("tick_costs", {})
    if not isinstance(tick_costs, Mapping):
        return {"verdict": "INVALID", "reason": "missing W1/K1 tick_costs"}
    cost_keys = (
        "state_copy_wall_ns",
        "state_copy_thread_cpu_ns",
        "state_copy_bytes",
        "publish_wall_ns",
        "publish_thread_cpu_ns",
        "poll_wall_ns",
        "poll_thread_cpu_ns",
        "tick_admitted",
        "tick_completed",
        "tick_overflow",
    )
    for key in cost_keys:
        series = tick_costs.get(key)
        if (
            not isinstance(series, list)
            or len(series) != expected_ticks
            or any(type(v) is not int or v < 0 for v in series)
        ):
            return {"verdict": "INVALID", "reason": f"malformed W1/K1 tick series: {key}"}

    cold_records = worker_data.get("cold_records")
    warm_records = worker_data.get("warm_records")
    if not isinstance(cold_records, list) or not isinstance(warm_records, list):
        return {"verdict": "INVALID", "reason": "missing cold/warm worker records"}
    if len(cold_records) != counts["prewarm_completed_count"] or len(cold_records) == 0:
        return {"verdict": "INVALID", "reason": "missing or mismatched cold prewarm records"}
    if len(warm_records) != counts["completed_count"]:
        return {"verdict": "INVALID", "reason": "mismatched warm completed record count"}
    for record in [*cold_records, *warm_records]:
        if (
            not isinstance(record, Mapping)
            or record.get("action_applied") is not False
            or record.get("reference_match") is not True
            or type(record.get("worker_service_ns")) is not int
            or record["worker_service_ns"] < 0
            or type(record.get("S_ns")) is not int
            or record["S_ns"] < record["worker_service_ns"]
        ):
            return {"verdict": "INVALID", "reason": "invalid worker reply record"}

    # Check native_v8 kernel execution counters across all completed records.
    native_totals = {
        "mode": "kernel",
        "calls": 0,
        "kernel_rows": 0,
        "stock_rows": 0,
        "noncandidate_rows": 0,
        "fallback_rows": 0,
    }
    for record in [*cold_records, *warm_records]:
        nv8 = record.get("native_v8", {})
        if not isinstance(nv8, Mapping) or nv8.get("mode") != "kernel":
            return {"verdict": "INVALID", "reason": "worker native_v8 mode is not kernel"}
        for key in ("calls", "kernel_rows", "stock_rows", "noncandidate_rows", "fallback_rows"):
            val = nv8.get(key)
            if type(val) is not int or val < 0:
                return {"verdict": "INVALID", "reason": f"invalid native_v8 counter {key}"}
            native_totals[key] += val
    if (
        native_totals["stock_rows"] != 0
        or native_totals["noncandidate_rows"] != 0
        or native_totals["fallback_rows"] != 0
    ):
        return {"verdict": "INVALID", "reason": "worker native_v8 fell back to stock rows"}

    state_copy_wall = tick_costs["state_copy_wall_ns"]
    state_copy_cpu = tick_costs["state_copy_thread_cpu_ns"]
    state_copy_bytes = tick_costs["state_copy_bytes"]
    publish_wall = tick_costs["publish_wall_ns"]
    publish_cpu = tick_costs["publish_thread_cpu_ns"]
    poll_wall = tick_costs["poll_wall_ns"]
    poll_cpu = tick_costs["poll_thread_cpu_ns"]
    admitted_flags = tick_costs["tick_admitted"]
    completed_flags = tick_costs["tick_completed"]

    admitted_indices = [i for i, flag in enumerate(admitted_flags) if flag == 1]
    completed_indices = [i for i, flag in enumerate(completed_flags) if flag == 1]

    combined_wall = [
        state_copy_wall[i] + publish_wall[i] + poll_wall[i] for i in range(expected_ticks)
    ]
    combined_cpu = [state_copy_cpu[i] + publish_cpu[i] + poll_cpu[i] for i in range(expected_ticks)]

    cold_service_samples = [int(r["worker_service_ns"]) for r in cold_records]
    warm_service_samples = [int(r["worker_service_ns"]) for r in warm_records]
    cold_s_samples = [int(r["S_ns"]) for r in cold_records]
    warm_s_samples = [int(r["S_ns"]) for r in warm_records]
    cold_cpu_samples = [int(r.get("worker_thread_cpu_ns", 0)) for r in cold_records]
    warm_cpu_samples = [int(r.get("worker_thread_cpu_ns", 0)) for r in warm_records]
    worker_rss_samples = [
        int(r["worker_rss_bytes"])
        for r in [*cold_records, *warm_records]
        if int(r.get("worker_rss_bytes", 0)) > 0
    ]

    fixture_ids = list(
        worker_data.get("fixture_ids") or sorted({r["fixture_id"] for r in cold_records})
    )
    per_fixture: dict[str, Any] = {}
    for fid in fixture_ids:
        f_cold = [r for r in cold_records if r["fixture_id"] == fid]
        f_warm = [r for r in warm_records if r["fixture_id"] == fid]
        per_fixture[str(fid)] = {
            "cold_count": len(f_cold),
            "warm_count": len(f_warm),
            "cold_worker_service_ns": _distribution([int(r["worker_service_ns"]) for r in f_cold]),
            "warm_worker_service_ns": _distribution([int(r["worker_service_ns"]) for r in f_warm]),
            "cold_s_ns": _distribution([int(r["S_ns"]) for r in f_cold]),
            "warm_s_ns": _distribution([int(r["S_ns"]) for r in f_warm]),
            "chosen_actions": sorted({int(r["chosen_action"]) for r in [*f_cold, *f_warm]}),
            "reference_action": (
                f_cold[0]["reference_action"]
                if f_cold
                else (f_warm[0]["reference_action"] if f_warm else None)
            ),
        }

    rss = worker_data.get("rss", {})
    if not isinstance(rss, Mapping):
        return {"verdict": "INVALID", "reason": "missing W1/K1 rss summary"}

    warm_s_dist = _distribution(warm_s_samples)
    warm_service_dist = _distribution(warm_service_samples)
    cold_s_dist = _distribution(cold_s_samples)
    cold_service_dist = _distribution(cold_service_samples)

    # Require warm completions on full/multi-second active runs (>= 24 ticks = 1.0s).
    if worker_mode == "active" and expected_ticks >= 24 and not warm_records:
        return {
            "verdict": "INCOMPLETE",
            "reason": "no warm worker completions observed during active measured window",
        }

    warm_s_p95 = warm_s_dist["p95"]
    warm_s_pass = warm_s_p95 is None or float(warm_s_p95) <= WARM_S_P95_LIMIT_NS
    verdict = "PASS" if warm_s_pass else "FAIL"

    return {
        "verdict": verdict,
        "worker_mode": worker_mode,
        "workers": 1,
        "max_pending": 1,
        "warm_s_p95_limit_ns": WARM_S_P95_LIMIT_NS,
        "warm_s_p95_pass": warm_s_pass,
        "service_latency": {
            "cold_worker_service_ns": cold_service_dist,
            "warm_worker_service_ns": warm_service_dist,
            "cold_worker_thread_cpu_ns": _distribution(cold_cpu_samples),
            "warm_worker_thread_cpu_ns": _distribution(warm_cpu_samples),
            "cold_s_ns": cold_s_dist,
            "warm_s_ns": warm_s_dist,
            "per_fixture": per_fixture,
        },
        "tick_ipc_costs": {
            "state_copy_wall_ns": _distribution(state_copy_wall),
            "state_copy_thread_cpu_ns": _distribution(state_copy_cpu),
            "state_copy_admitted_wall_ns": _distribution(
                [state_copy_wall[i] for i in admitted_indices]
            ),
            "state_copy_admitted_thread_cpu_ns": _distribution(
                [state_copy_cpu[i] for i in admitted_indices]
            ),
            "state_copy_bytes": _distribution(state_copy_bytes),
            "publish_wall_ns": _distribution(publish_wall),
            "publish_thread_cpu_ns": _distribution(publish_cpu),
            "publish_admitted_wall_ns": _distribution([publish_wall[i] for i in admitted_indices]),
            "publish_admitted_thread_cpu_ns": _distribution(
                [publish_cpu[i] for i in admitted_indices]
            ),
            "poll_wall_ns": _distribution(poll_wall),
            "poll_thread_cpu_ns": _distribution(poll_cpu),
            "poll_hit_wall_ns": _distribution([poll_wall[i] for i in completed_indices]),
            "poll_hit_thread_cpu_ns": _distribution([poll_cpu[i] for i in completed_indices]),
            "submission_transfer_poll_wall_ns": _distribution(combined_wall),
            "submission_transfer_poll_thread_cpu_ns": _distribution(combined_cpu),
        },
        "rss": {
            **dict(rss),
            "worker_rss_bytes_samples": _distribution(worker_rss_samples),
        },
        "counts": dict(counts),
        "native_v8_totals": native_totals,
    }
