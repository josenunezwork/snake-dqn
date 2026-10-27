"""Fresh-process runner for the served tick diagnostic.

This is deliberately a diagnostic transport, not a server split: the child owns
one real Watch6 session and returns only a bounded immutable report after its
workload completes.  Its Pipe timings are charged to the result.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Any, Mapping


def _child(connection: Any, spec: Mapping[str, Any]) -> None:
    """Run one arm in a fresh interpreter and publish one finite payload."""
    from research.bl_served_search.served_tick_probe import run_inline_arm

    try:
        payload = run_inline_arm(dict(spec), fresh_process=True)
        # Measure only the bounded final report send. The parent's receive
        # timestamp is sent afterwards so neither value silently includes the
        # whole game lifetime.
        started_wall, started_cpu = time.perf_counter_ns(), time.thread_time_ns()
        connection.send({"ok": True, "payload": payload})
        connection.send(
            {
                "kind": "send_metrics",
                "wall_ns": time.perf_counter_ns() - started_wall,
                "thread_cpu_ns": time.thread_time_ns() - started_cpu,
            }
        )
    except BaseException as exc:  # surface child failures with their class
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def run_isolated_arm(spec: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    """Run ``spec`` in a spawned process, charging bounded report IPC explicitly."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(child, dict(spec)), daemon=False)
    lifecycle_started = time.perf_counter_ns()
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=10)
            raise TimeoutError(f"isolated game arm exceeded {timeout_seconds}s")
        recv_started, recv_cpu = time.perf_counter_ns(), time.thread_time_ns()
        message = parent.recv()
        recv_wall, recv_cpu_elapsed = (
            time.perf_counter_ns() - recv_started,
            time.thread_time_ns() - recv_cpu,
        )
        if not message.get("ok"):
            raise RuntimeError(f"isolated game child failed: {message.get('error')}")
        if not parent.poll(10):
            raise RuntimeError("isolated game child omitted send metrics")
        send_metrics = parent.recv()
    finally:
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    if process.exitcode not in (0, None):
        raise RuntimeError(f"isolated game child exit={process.exitcode}")
    payload = dict(message["payload"])
    payload["ipc"] = {
        "child_report_send_wall_ns": int(send_metrics["wall_ns"]),
        "child_report_send_thread_cpu_ns": int(send_metrics["thread_cpu_ns"]),
        "parent_report_receive_wall_ns": recv_wall,
        "parent_report_receive_thread_cpu_ns": recv_cpu_elapsed,
        "child_lifecycle_wall_ns": time.perf_counter_ns() - lifecycle_started,
        "transport": "spawn-pipe-bounded-final-report",
        "dto_renderer_split": "NOT_MEASURED",
    }
    return payload
