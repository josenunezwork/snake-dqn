"""First, outcome-free timing package for the real mixed Watch6 path.

No production serving path imports this module.  It measures the actual
``research.bl_k1.m1.new_watch`` constructor under a 24 fps absolute clock and
exports only after a cell has stopped; recorder rows never write to disk on a
tick.  A later reducer owns bar interpretation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from research.bl_paced_identity.record import export_json, read_export

SCHEMA = "served-tick-diagnostic/v1"
FPS = 24
WARMUP_TICKS = 200
MEASURED_TICKS = 2400
FRAME_NS = 1_000_000_000 // FPS
UPDATE_BAR_MS = 11.688
_RUNTIME_TIMING_FIELDS = frozenset(
    {
        "apply_seconds_total",
        "apply_seconds_max",
        "head_seconds_total",
        "head_seconds_max",
        "area_eval_seconds_total",
        "area_eval_seconds_max",
        "mean_apply_seconds",
        "reference_seconds_total",
        "reference_seconds_max",
        "mean_area_eval_seconds_per_evaluation",
        "search_seconds_total",
        "search_seconds_max",
        "mean_search_seconds",
        "fallback_seconds_total",
        "fallback_seconds_max",
        "mean_fallback_seconds",
    }
)


def _is_runtime_timing_field(name: str) -> bool:
    """Exclude diagnostic duration counters, never decisions/carry/action fields."""
    return name in _RUNTIME_TIMING_FIELDS


def _canonical(value: Any, seen: set[int] | None = None) -> Any:
    """Make a finite digest input including mutable game/RNG/carry state.

    Policies and locks are represented by their identity-free public state; this
    intentionally avoids serialising models while retaining every mutable field
    on the session/game/snakes and their wrappers.
    """
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return {"path": str(value)}
    marker = id(value)
    if marker in seen:
        return {"cycle": type(value).__name__}
    if isinstance(value, bytes):
        return {"bytes": hashlib.sha256(value).hexdigest(), "size": len(value)}
    # NumPy and torch values need bytes/dtype/shape, not repr()/tolist() which
    # can hide dtype drift and can allocate an enormous nested Python graph.
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            return _canonical(value.detach().cpu().numpy(), seen)
        except Exception:
            pass
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        try:
            return {
                "array": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        except Exception:
            pass
    if isinstance(value, Mapping):
        seen.add(marker)
        result = {
            str(key): _canonical(item, seen)
            for key, item in sorted(value.items(), key=lambda x: repr(x[0]))
            if not _is_runtime_timing_field(str(key))
        }
        seen.remove(marker)
        return result
    if isinstance(value, (list, tuple, deque, set, frozenset)):
        seen.add(marker)
        items = [_canonical(item, seen) for item in value]
        seen.remove(marker)
        return items if not isinstance(value, (set, frozenset)) else sorted(items, key=repr)
    if hasattr(value, "getstate"):
        try:
            return {"rng": _canonical(value.getstate(), seen)}
        except Exception:
            pass
    slots = []
    for cls in type(value).__mro__:
        names = getattr(cls, "__slots__", ())
        slots.extend((names,) if isinstance(names, str) else names)
    if slots:
        seen.add(marker)
        result = {
            "slots": type(value).__qualname__,
            "fields": {
                name: _canonical(getattr(value, name), seen)
                for name in sorted(set(slots))
                if hasattr(value, name) and not _is_runtime_timing_field(name)
            },
        }
        seen.remove(marker)
        return result
    if hasattr(value, "__dict__"):
        seen.add(marker)
        fields = {
            key: item
            for key, item in vars(value).items()
            if key
            not in {
                "_lock",
                "dqn",
                "target_dqn",
                "optimizer",
                "device",
                "net",
                "_run_started",
                "session_id",
            }
            and not _is_runtime_timing_field(key)
            and not callable(item)
        }
        result = {"object": type(value).__qualname__, "fields": _canonical(fields, seen)}
        seen.remove(marker)
        return result
    # Address-bearing repr strings are not semantic state. Unknown values fail
    # closed instead of making a cross-process equality claim nondeterministic.
    raise TypeError(
        f"unsupported semantic state type: {type(value).__module__}.{type(value).__qualname__}"
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def session_state_digest(session: Any) -> str:
    """Digest complete mutable game/session state, including per-policy carry/RNG."""
    import numpy as np
    import torch

    from src.core.game_config import get_config

    return _digest(
        {
            "session": session,
            "game": session.game,
            "snake_policies": [getattr(snake, "policy", None) for snake in session.game.snakes],
            "python_random": __import__("random").getstate(),
            "numpy_random": np.random.get_state(),
            "torch_cpu_rng": torch.get_rng_state(),
            "runtime_config": get_config(),
        }
    )


def dto_digest(dto: Mapping[str, Any]) -> dict[str, str]:
    """Digest each detached DTO section for actionable cross-process parity."""
    result: dict[str, str] = {}
    for key, value in sorted(dto.items()):
        if isinstance(value, Mapping):
            for child, nested in sorted(value.items()):
                result[f"{key}.{child}"] = _digest(nested)
        else:
            result[str(key)] = _digest(value)
    return result


def _identity(session: Any, roster: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    source_paths = [
        root / "research/bl_k1/m1.py",
        root / "research/bl_served_search/served_tick_probe.py",
        root / "research/bl_served_search/isolated_game.py",
        root / "research/bl_served_search/tick_trace.py",
        root / "web/backend/session.py",
        root / "web/backend/serialize.py",
        root / "web/backend/ego2s_policy.py",
        root / "src/game/game_state.py",
        root / "configs/free_space_v2.yaml",
        root / "configs/mechanics_v2.yaml",
    ]
    source_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
        if path.is_file()
    }
    return {
        "roster": dict(roster),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "env_threads": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
        "pid": os.getpid(),
        "gc_enabled": gc.isenabled(),
        "session_type": type(session).__qualname__,
        "source_sha256": source_hashes,
        "release_monitoring": getattr(session, "_release_monitoring", None),
        "release_monitoring_note": (
            "T15 constructor disables telemetry; endpoint is actual GameState.update"
        ),
    }


def _sleep_until(target_ns: int) -> int:
    now = time.perf_counter_ns()
    delay = target_ns - now
    if delay > 0:
        time.sleep(delay / 1_000_000_000)
    return time.perf_counter_ns()


@contextmanager
def _legacy_numpy_seed_projection(seed: int) -> Iterator[None]:
    """Preserve uint64 world identity while adapting only NumPy's legacy API."""
    import random
    from unittest.mock import patch

    import numpy as np
    import torch

    from research.t15_mixed_roster_20261017 import episodes

    def set_seed(value: int) -> None:
        if int(value) != int(seed):
            raise ValueError("probe constructor received a foreign world seed")
        random.seed(value)
        np.random.seed(value & 0xFFFFFFFF)
        torch.manual_seed(value)

    with patch.object(episodes, "set_world_seed", set_seed):
        yield


def new_probe_watch(seed: int) -> tuple[Any, dict[str, Any]]:
    """Construct actual Watch6 with a documented NumPy-only uint32 projection."""
    from research.bl_k1.m1 import new_watch

    with _legacy_numpy_seed_projection(seed):
        return new_watch(seed)


def run_inline_arm(spec: Mapping[str, Any], fresh_process: bool = False) -> dict[str, Any]:
    """Run timer-only or real Watch6 step/snapshot workload in this process."""
    from research.bl_served_search.tick_trace import TraceRecorder

    arm = str(spec["arm"])
    measured = int(spec.get("measured_ticks", MEASURED_TICKS))
    warmup = int(spec.get("warmup_ticks", WARMUP_TICKS))
    seed = int(spec["seed"])
    effective_seed = seed & 0xFFFFFFFF
    trace_enabled = bool(spec.get("trace_enabled", True))
    recorder = TraceRecorder(capacity=warmup + measured + 8) if trace_enabled else None
    session = None
    roster: dict[str, Any] = {}
    if arm != "timer":
        from research.t1_commitment_20261015.replay import configure_cpu

        configure_cpu()
        session, roster = new_probe_watch(seed)
    try:
        if session is not None:
            for _ in range(warmup):
                session.step()
                session.snapshot()
        # Full semantic digests are deliberately outside the critical measurement
        # loop.  Per-tick action capture is small, preallocated evidence only.
        initial_state = session_state_digest(session) if session is not None else None
        actions: list[Any] = [None] * measured
        processing_wall_ns: list[int] = [0] * measured
        processing_thread_cpu_ns: list[int] = [0] * measured
        dto: Mapping[str, Any] | None = None
        with (
            recorder.attach(session)
            if session is not None and recorder is not None
            else _nullcontext()
        ):
            epoch = time.perf_counter_ns()
            for index in range(measured):
                target_ns = epoch + (index * 1_000_000_000) // FPS
                wake_ns = _sleep_until(target_ns)
                processing_started = time.perf_counter_ns()
                processing_cpu = time.thread_time_ns()
                if recorder is not None:
                    recorder.begin_tick(index, target_ns, wake_ns)
                try:
                    if session is not None:
                        session.step()
                        dto = session.snapshot()
                        actions[index] = [
                            getattr(snake, "last_action", None) for snake in session.game.snakes
                        ]
                        if recorder is not None:
                            recorder.capture_session_context(session)
                finally:
                    if recorder is not None:
                        recorder.end_tick(time.perf_counter_ns())
                    processing_wall_ns[index] = time.perf_counter_ns() - processing_started
                    processing_thread_cpu_ns[index] = time.thread_time_ns() - processing_cpu
        final_state = session_state_digest(session) if session is not None else None
        final_dto = dto_digest(dto) if dto is not None else None
        return {
            "schema": SCHEMA,
            "arm": arm,
            "seed": seed,
            "effective_seed_uint32": effective_seed,
            "fresh_process": fresh_process,
            "fps": FPS,
            "warmup_ticks": warmup,
            "measured_ticks": measured,
            "identity": _identity(session, roster) if session is not None else {"timer_only": True},
            "trace": recorder.rows() if recorder is not None else [],
            "state_digest": {"initial": initial_state, "final": final_state},
            "dto_digest": {"final": final_dto},
            "actions": actions,
            "processing_wall_ns": processing_wall_ns,
            "processing_thread_cpu_ns": processing_thread_cpu_ns,
            "gc": {"enabled_after": gc.isenabled(), "counts": list(gc.get_count())},
            "client_receive": "NOT_MEASURED_NO_WEBSOCKET_CLIENT",
            "frame_send": "NOT_MEASURED_NO_WEBSOCKET_CLIENT",
            "export_copy": "NOT_MEASURED_W0_NO_PROCESS_DTO_SPLIT",
            "ipc": "NOT_MEASURED_W0_NO_PROCESS_DTO_SPLIT",
        }
    finally:
        if session is not None:
            session.close(wait=True)


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: Any) -> None:
        return None


def _verify_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("state_digest", "dto_digest", "actions")
    mismatches = [
        field
        for field in fields
        if not left.get(field) or not right.get(field) or left[field] != right[field]
    ]
    return {"fields": list(fields), "pass": not mismatches, "mismatches": mismatches}


def source_binding(require_clean: bool = True) -> dict[str, Any]:
    """Bind the complete tracked tree plus the small probe closure off the clock."""
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if require_clean and dirty:
        raise ValueError("timing requires a committed, clean tracked worktree")
    if require_clean:
        git(
            "ls-files",
            "--error-unmatch",
            "research/bl_served_search/served_tick_probe.py",
            "research/bl_served_search/served_tick_manifest.json",
        )
    paths = [
        *Path(__file__).parent.glob("*tick*"),
        Path(__file__).with_name("probe_host.py"),
        Path(__file__).with_name("host_load.py"),
        Path(__file__).with_name("background_report.py"),
        Path(__file__).with_name("HOST_LOAD_AMENDMENT_20261027.md"),
        Path(__file__).with_name("production_probe.py"),
        Path(__file__).with_name("isolated_game.py"),
        root / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json",
    ]
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_dirty": dirty,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))
            if path.is_file()
        },
    }


def _distribution(samples: list[int]) -> dict[str, float | int | None]:
    """Use the trace reducer's percentile convention, outside all timed loops."""
    import numpy as np

    if not samples:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(samples),
        **{f"p{q}": float(np.percentile(samples, q)) for q in (50, 95, 99)},
        "max": max(samples),
    }


def _production_summary(result: Mapping[str, Any], expected: int) -> dict[str, Any]:
    """Keep Hub's relative sleep and distinct worker clocks out of absolute bars."""
    from research.bl_served_search.tick_trace import summarize_breakdowns

    rows = result.get("trace", [])
    if len(rows) != expected:
        return {"verdict": "INCOMPLETE", "reason": "production exposure"}
    updates, step_cpu, snapshot_cpu, latencies, sends = [], [], [], [], []
    received = [
        row for row in result.get("client_receive", []) if row.get("source") == "hub_broadcast"
    ]
    sent = list(result.get("frame_send", []))
    warmup = result.get("warmup_ticks", 0)
    if (
        type(warmup) is not int
        or warmup < 0
        or len(received) != warmup + expected
        or len(sent) != warmup + expected
    ):
        return {"verdict": "INVALID", "reason": "production delivery exposure"}
    received, sent = received[warmup:], sent[warmup:]
    measured_receipts = []
    for index, row in enumerate(rows):
        try:
            update = row["spans"]["game_update"]
            duration = update["wall_ns"]
            if (
                row["tick"] != index
                or update["calls"] != 1
                or type(duration) is not int
                or duration < 0
                or not row["start_ns"]
                <= update["first_wall_start_ns"]
                <= update["last_wall_end_ns"]
                <= row["end_ns"]
            ):
                raise ValueError("update span")
            frame = row["game_age"]
            receipt, send = received[index], sent[index]
            receive = receipt["receive_ns"]
            if (
                receipt["frame"] != frame
                or send["frame"] != frame
                or send["status"] != "ok"
                or receive < send["start_ns"]
            ):
                raise ValueError("delivery timestamps")
            updates.append(duration)
            step_cpu.append(row["end_cpu_ns"] - row["start_cpu_ns"])
            snapshot_cpu.append(row["snapshot_end_cpu_ns"] - row["snapshot_start_cpu_ns"])
            latencies.append(receive - row["start_ns"])
            sends.append(receive - send["start_ns"])
            measured_receipts.append(receive)
        except (KeyError, TypeError, ValueError):
            return {"verdict": "INVALID", "reason": "production span/delivery", "tick": index}
    gaps = [b - a for a, b in zip(measured_receipts, measured_receipts[1:])]
    if any(value < 0 for values in (step_cpu, snapshot_cpu, gaps) for value in values):
        return {"verdict": "INVALID", "reason": "production CPU/cadence ordering"}
    update_summary = _distribution(updates)
    try:
        breakdowns = summarize_breakdowns(rows)
    except (KeyError, TypeError, ValueError) as exc:
        return {"verdict": "INVALID", "reason": f"production breakdowns: {exc}"}
    return {
        "verdict": "PASS" if update_summary["p95"] <= 11_688_000 else "FAIL",
        "absolute_overrun_applicable": False,
        "endpoint": "actual Hub processing plus full sleep; separate step/snapshot CPU clocks",
        "update_ns": update_summary,
        "step_thread_cpu_ns": _distribution(step_cpu),
        "snapshot_thread_cpu_ns": _distribution(snapshot_cpu),
        "client_frame_gap_ns": _distribution(gaps),
        "step_to_client_ns": _distribution(latencies),
        "send_to_client_ns": _distribution(sends),
        **breakdowns,
        "candidate_client_noninferiority": "NOT_TESTED: baseline endpoint only",
    }


def reduce_trace(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Use the trace owner's reducer; combine fixed cells without reinterpretation."""
    from research.bl_served_search.tick_trace import reduce_trace as reduce_one

    manifest = raw.get("manifest", {})
    try:
        preflight(manifest)
    except (ValueError, KeyError, TypeError) as exc:
        return {"verdict": "INVALID", "reason": f"manifest: {exc}"}
    cells = raw.get("cells", [])
    actual_order = [entry.get("cell") for entry in cells]
    if actual_order != manifest["order"]:
        return {
            "verdict": "INCOMPLETE" if len(cells) < len(manifest["order"]) else "INVALID",
            "reason": "fixed cell inventory/order differs",
            "observed_cells": len(cells),
        }
    binding = raw.get("source_before", {})
    if (
        binding != raw.get("source_after")
        or not binding.get("source_sha256")
        or binding.get("tracked_dirty") != ""
        or len(binding.get("commit", "")) != 40
    ):
        return {"verdict": "INVALID", "reason": "missing or changed source binding"}
    if raw.get("host", {}).get("status") != "COMPLETE" or (
        (raw.get("lease") or {}).get("monitor_status") == "INVALID"
    ):
        return {"verdict": "INVALID", "reason": "missing or contaminated host evidence"}
    classified = raw.get("load_policy") == "classified-v1"
    if raw.get("load_policy") != raw.get("host", {}).get("load_policy"):
        return {"verdict": "INVALID", "reason": "host load policy binding differs"}
    if raw.get("load_policy") not in (None, "classified-v1"):
        return {"verdict": "INVALID", "reason": "unsupported host load policy"}
    if classified and raw["host"].get("schema") != "served-tick-host/v2":
        return {"verdict": "INVALID", "reason": "classified host schema differs"}
    arms: list[dict[str, Any]] = []
    for cell in raw.get("cells", []):
        result = cell["result"]
        expected_seed = manifest["seeds"][cell["cell"]["seed_index"]]
        if (
            result.get("seed") != expected_seed
            or result.get("fps") != manifest["fps"]
            or result.get("warmup_ticks") != manifest["warmup_ticks"]
            or result.get("measured_ticks") != manifest["measured_ticks"]
        ):
            return {"verdict": "INVALID", "reason": "cell workload identity differs"}
        rows = list(cell["result"].get("trace", ()))
        covariates: dict[str, Any] = {}
        if classified:
            from research.bl_served_search.background_report import summarize_background

            try:
                covariates["background_load"] = summarize_background(
                    result, raw["host"], production=cell["cell"].get("arm") == "production"
                )
            except (KeyError, TypeError, ValueError) as exc:
                return {"verdict": "INVALID", "reason": f"background evidence: {exc}"}
        if cell["cell"].get("arm") == "production":
            arms.append(
                {
                    "cell": cell["cell"],
                    **_production_summary(cell["result"], int(manifest["measured_ticks"])),
                    **covariates,
                }
            )
            continue
        arm = str(cell["cell"].get("arm"))
        arms.append(
            {
                "cell": cell["cell"],
                **reduce_one(
                    rows,
                    int(raw["manifest"]["measured_ticks"]),
                    int(raw["manifest"]["fps"]),
                    require_update=arm != "timer",
                ),
                **covariates,
            }
        )
    pairs = []
    for seed_index in range(3):
        results = {
            cell["cell"]["arm"]: cell["result"]
            for cell in cells
            if cell["cell"]["seed_index"] == seed_index
        }
        pair = _verify_pair(results["inline"], results["isolated"])
        pairs.append({"seed_index": seed_index, **pair})
    observer = raw.get("observer_control", {})
    observer_arms = observer.get("arms", [])
    observer_valid = (
        observer.get("ticks") == manifest["observer_ticks"]
        and [entry.get("name") for entry in observer_arms]
        == manifest["instrumented_observer_order"]
    )
    if not observer_valid or not all(pair["pass"] for pair in pairs):
        return {
            "verdict": "INVALID",
            "reason": "missing observer or state/action/DTO parity",
            "arms": arms,
            "parity": pairs,
        }
    observer_pair = _verify_pair(observer_arms[0]["result"], observer_arms[1]["result"])
    if not observer_pair["pass"]:
        return {"verdict": "INVALID", "reason": "observer semantic parity", "parity": observer_pair}
    overhead = []
    for entry in observer_arms:
        durations = entry["result"].get("processing_wall_ns", [])
        cpu = entry["result"].get("processing_thread_cpu_ns", [])
        if (
            len(durations) != manifest["observer_ticks"]
            or len(cpu) != len(durations)
            or any(type(value) is not int or value < 0 for value in [*durations, *cpu])
        ):
            return {"verdict": "INVALID", "reason": "missing recorder overhead exposure"}
        overhead.append(
            {
                "name": entry["name"],
                "processing_wall_ns": _distribution(durations),
                "processing_thread_cpu_ns": _distribution(cpu),
            }
        )
    overhead_delta = [
        a - b
        for a, b in zip(
            observer_arms[1]["result"]["processing_wall_ns"],
            observer_arms[0]["result"]["processing_wall_ns"],
        )
    ]
    verdicts = [arm.get("verdict") for arm in arms if "verdict" in arm]
    verdict = (
        "INVALID"
        if "INVALID" in verdicts
        else (
            "INCOMPLETE"
            if "INCOMPLETE" in verdicts
            else (
                "FAIL"
                if "FAIL" in verdicts
                else (
                    "PASS"
                    if verdicts and all(value == "PASS" for value in verdicts)
                    else "INCOMPLETE"
                )
            )
        )
    )
    return {
        "schema": f"{SCHEMA}-reducer",
        "arms": arms,
        "verdict": verdict,
        "parity": pairs,
        "observer_parity": observer_pair,
        "recorder_overhead": overhead,
        "paired_recorder_wall_delta_ns": _distribution(overhead_delta),
        "scope": "engineering timing only; no release authority",
    }


@contextmanager
def _poll_host(monitor: Any, interval_seconds: float = 1.0) -> Iterator[None]:
    """Poll every arm on one off-tick parent thread, without serializing the monitor."""
    if monitor is None:
        yield
        return
    stopped = threading.Event()
    errors: list[BaseException] = []

    def poll() -> None:
        while not stopped.wait(interval_seconds):
            try:
                monitor.check()
            except BaseException as exc:
                errors.append(exc)
                return

    monitor.check()
    thread = threading.Thread(target=poll, name="served-tick-host-poll", daemon=True)
    thread.start()
    try:
        yield
        if errors:
            raise RuntimeError("host polling failed during panel") from errors[0]
        monitor.check()
    finally:
        stopped.set()
        thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError("host polling thread failed to stop")


def run_panel(
    manifest: Mapping[str, Any], output: Path, host_monitor: Any = None
) -> dict[str, Any]:
    """Keep host health polling uniform across inline, child and observer workloads."""
    with _poll_host(host_monitor):
        return _run_panel(manifest, output, host_monitor)


def _run_panel(
    manifest: Mapping[str, Any], output: Path, host_monitor: Any = None
) -> dict[str, Any]:
    """Execute the fixed W0 panel in a lease; exporting belongs after its final sample."""
    from research.bl_served_search.isolated_game import run_isolated_arm

    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected served tick manifest schema")
    deadline = time.monotonic() + float(manifest["timeout_seconds"])
    cells: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(manifest["order"]):
        if host_monitor is not None:
            host_monitor.check()
        indexed_seed = (
            manifest["seeds"][int(cell["seed_index"])] if "seed_index" in cell else cell["seed"]
        )
        spec = {
            **cell,
            "seed": indexed_seed,
            "warmup_ticks": manifest["warmup_ticks"],
            "measured_ticks": manifest["measured_ticks"],
        }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("whole diagnostic cap exhausted before next cell")
        if cell["arm"] == "isolated":
            # A monitor object contains multiprocessing primitives and is not a
            # child input; parent/child transport checks are bounded separately.
            result = run_isolated_arm(spec, remaining)
        elif cell["arm"] == "production":
            from research.bl_served_search.production_probe import run_production_arm

            result = run_production_arm(spec, timeout_seconds=remaining)
        else:
            result = run_inline_arm(spec)
        cells.append({"cell": dict(cell), "result": result})
        export_json(output / f"cell-{cell_index:02d}.json", cells[-1])
        print(
            json.dumps(
                {
                    "completed_cells": cell_index + 1,
                    "total_cells": len(manifest["order"]),
                    "arm": cell["arm"],
                    "seed_index": cell.get("seed_index"),
                }
            ),
            flush=True,
        )
        if host_monitor is not None:
            host_monitor.check()
    raw = {
        "schema": SCHEMA,
        "manifest": dict(manifest),
        "cells": cells,
        "scope": "W0 bare Watch6 and production Hub baseline; no search or renderer split",
        "lease": dict(host_monitor.receipt) if host_monitor is not None else None,
    }
    # Pair only same seed/current-vs-fresh process.  A mismatch is evidence, never discarded.
    pairs = []
    for seed in manifest["seeds"]:
        inline = next(
            (
                x["result"]
                for x in cells
                if x["result"].get("seed") == seed and x["cell"]["arm"] == "inline"
            ),
            None,
        )
        isolated = next(
            (
                x["result"]
                for x in cells
                if x["result"].get("seed") == seed and x["cell"]["arm"] == "isolated"
            ),
            None,
        )
        if inline is not None and isolated is not None:
            pairs.append({"seed": seed, "parity": _verify_pair(inline, isolated)})
    raw["parity"] = pairs
    # The recorder-control is fixed, short, and separate from timing bars. It
    # bounds instrumentation semantic influence without treating its tail as a
    # qualification exposure.
    observer_seed = int(manifest["observer_seed"])
    observer_ticks = int(manifest.get("observer_ticks", 120))
    observer = []
    for name in manifest.get("instrumented_observer_order", ("uninstrumented", "instrumented")):
        if time.monotonic() >= deadline:
            raise TimeoutError("whole diagnostic cap exhausted before observer control")
        if host_monitor is not None:
            host_monitor.check()
        observer.append(
            {
                "name": name,
                "result": run_inline_arm(
                    {
                        "arm": "inline",
                        "seed": observer_seed,
                        "warmup_ticks": 0,
                        "measured_ticks": observer_ticks,
                        "trace_enabled": name == "instrumented",
                    }
                ),
            }
        )
    if len(observer) == 2:
        raw["observer_control"] = {
            "ticks": observer_ticks,
            "order": [entry["name"] for entry in observer],
            "semantic_parity": _verify_pair(observer[0]["result"], observer[1]["result"]),
            "arms": observer,
        }
        export_json(output / "observer-control.json", raw["observer_control"])
    return raw


def preflight(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Non-mutating static validation; deliberately does not import torch/session."""
    required = {
        "schema",
        "fps",
        "warmup_ticks",
        "measured_ticks",
        "seeds",
        "arms",
        "order",
        "workers",
        "max_pending",
    }
    missing = sorted(required - set(manifest))
    if (
        missing
        or manifest.get("schema") != SCHEMA
        or manifest.get("workers") != 0
        or manifest.get("max_pending") != 0
    ):
        raise ValueError(f"invalid W0 served-tick manifest missing={missing}")
    if (
        int(manifest["fps"]) != FPS
        or int(manifest["warmup_ticks"]) != WARMUP_TICKS
        or int(manifest["measured_ticks"]) != MEASURED_TICKS
    ):
        raise ValueError("first package requires frozen 24fps/200-warmup/2400-measured exposure")
    committed = json.loads(Path(__file__).with_name("served_tick_manifest.json").read_text())
    frozen = (
        "arms",
        "order",
        "base_timeout_seconds",
        "timeout_seconds",
        "observer_ticks",
        "instrumented_observer_order",
    )
    if any(manifest.get(field) != committed.get(field) for field in frozen):
        raise ValueError("served-tick manifest differs from the committed fixed panel")
    seeds = [int(seed) for seed in manifest["seeds"]]
    observer_seed = int(manifest.get("observer_seed", -1))
    if len(seeds) != 3 or len(set([*seeds, observer_seed])) != 4:
        raise ValueError("require three distinct panel seeds plus one observer-control seed")
    from tools.sandbox.registry import load_registry, validate_input

    registry = load_registry()
    indices = list(range(4))
    validate_input(
        {
            "kind": "worlds",
            "study": "served-tick-20261026",
            "indices": indices,
            "seeds": [*seeds, observer_seed],
        },
        registry,
    )
    excluded32 = {int(seed) & 0xFFFFFFFF for seed in registry["excluded_seeds"]}
    collisions32 = sorted(
        seed for seed in [*seeds, observer_seed] if (seed & 0xFFFFFFFF) in excluded32
    )
    if collisions32:
        raise ValueError(f"uint32-normalized seed collides with registry exclusion: {collisions32}")
    registry_path = Path(
        __import__("tools.sandbox.registry", fromlist=["DEFAULT_REGISTRY"]).DEFAULT_REGISTRY
    )
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    if (
        manifest.get("registry_path") != "tools/sandbox/SANDBOX_INPUT_REGISTRY.json"
        or manifest.get("registry_sha256") != registry_sha
    ):
        raise ValueError("manifest registry binding differs from current protected-input registry")
    return {
        "schema": SCHEMA,
        "status": "PREFLIGHT_OK_NOT_TIMING",
        "cells": len(manifest["order"]),
        "seed_registry_sha256": registry_sha,
        "effective_numpy_seeds": [seed & 0xFFFFFFFF for seed in [*seeds, observer_seed]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("served_tick_manifest.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reduce", type=Path)
    parser.add_argument(
        "--classified-host-load",
        action="store_true",
        help="record macOS/desktop background CPU covariates; abort sustained >150%% for >60s",
    )
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    if args.reduce is not None:
        if args.output is None:
            parser.error("--reduce requires a fresh --output report directory")
        args.output.mkdir(parents=True, exist_ok=False)
        raw = read_export(args.reduce)
        # Re-read the independent host triplet; embedded status alone is not proof.
        host_path = args.reduce.parent / "host.json"
        if read_export(host_path) != raw.get("host"):
            raise ValueError("raw host evidence differs from verified host export")
        report = reduce_trace(raw)
        export_json(args.output / "report.json", report)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.check:
        from research.bl_served_search.probe_host import preflight as host_preflight

        result = {
            "probe": preflight(manifest),
            "host": host_preflight(classified=args.classified_host_load),
            "source": source_binding(require_clean=False),
        }
        print(json.dumps(result, sort_keys=True))
        if not result["host"]["admitted"]:
            return 2
        return 0
    if args.output is None:
        parser.error("--output is required unless --check")
    # Refuse a changed or unadmitted manifest before creating a directory,
    # leasing the Mac, importing the product, or starting a fresh process.
    preflight(manifest)
    binding = source_binding(require_clean=not args.smoke)
    args.output.mkdir(parents=True, exist_ok=False)
    from research.bl_served_search.probe_host import guarded_run

    try:
        if args.smoke:
            # Functional only: no lease/quiet requirement and no paced timing
            # conclusion.  It still requires the launcher's CPU/nice environment.
            from research.bl_served_search.isolated_game import run_isolated_arm
            from research.bl_served_search.probe_host import preflight as host_preflight

            cell = {
                "arm": "inline",
                "seed": manifest["seeds"][0],
                "warmup_ticks": 1,
                "measured_ticks": 2,
            }
            inline = run_inline_arm(cell)
            isolated = run_isolated_arm(cell, timeout_seconds=120)
            result = {
                "schema": f"{SCHEMA}-smoke/v1",
                "status": "SMOKE_OK_NOT_TIMING",
                "host_readonly": host_preflight(classified=args.classified_host_load),
                "inline": inline,
                "isolated": isolated,
                "parity": _verify_pair(inline, isolated),
            }
            if not result["parity"]["pass"]:
                result["status"] = "INVALID_SEMANTIC_PARITY"
            export_json(args.output / "smoke.json", result)
            return 0 if result["parity"]["pass"] else 2
        guard_options = {"classified": True} if args.classified_host_load else {}
        raw = guarded_run(
            args.output, lambda monitor: run_panel(manifest, args.output, monitor), **guard_options
        )
        # guarded_run returned only after HostMonitor wrote and sealed its final
        # sample.  Verify that immutable evidence before publishing any verdict.
        host = read_export(args.output / "host.json")
        raw["host"] = host
        if args.classified_host_load:
            raw["load_policy"] = "classified-v1"
        raw["source_before"] = binding
        raw["source_after"] = source_binding()
        raw["argv"] = list(sys.argv)
        raw["interpreter"] = {
            "executable": sys.executable,
            "version": sys.version,
            "os": platform.platform(),
        }
        export_json(args.output / "raw.json", raw)
        report = reduce_trace(raw)
        report["host_status"] = host.get("status")
        if host.get("status") != "COMPLETE":
            report["verdict"] = "INVALID"
        export_json(args.output / "report.json", report)
        print(
            json.dumps({"report": str(args.output / "report.json"), "verdict": report["verdict"]}),
            flush=True,
        )
        return 0 if report["verdict"] == "PASS" else 2
    except BaseException as exc:
        attempt = {
            "schema": f"{SCHEMA}-attempt/v1",
            "status": "INVALID",
            "reason": f"{type(exc).__name__}: {exc}",
            "manifest": manifest,
            "load_policy": "classified-v1" if args.classified_host_load else "legacy",
            "source_before": binding,
            "completed_cell_files": sorted(path.name for path in args.output.glob("cell-*.json")),
        }
        try:
            export_json(args.output / "attempt.json", attempt)
        except FileExistsError:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
