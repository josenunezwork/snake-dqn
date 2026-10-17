"""Instrument the failed Watch12 pacing path without changing serving behavior.

This is diagnostic-only.  It drives the same ``GameSession.step(); snapshot()``
order as ``Hub.engine_loop``, performs Hub's actual ``json.dumps(message)``
serialization, and sleeps a full requested interval after work.  Patches are
temporary timing wrappers around already-selected functions; no policy result,
forward mode, or serving source is changed.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import statistics
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from research.t12_b8_serving_20261017 import measurements  # noqa: E402

DEFAULT_OUTPUT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t12b-check4-20261017/diagnosis-v1.json"
)
DIAGNOSTIC_SEEDS = {24: 2026101724, 17: 2026101717, "websocket_24": 2026101725}
WARMUP_TICKS = 200
PACED_TICKS = 120
WEBSOCKET_OUTPUT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t12b-check4-20261017/"
    "diagnosis-websocket-v1.json"
)
WEBSOCKET_SEED = DIAGNOSTIC_SEEDS["websocket_24"]


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = (len(ordered) - 1) * quantile
    low, high = int(index), int(-(-index // 1))
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (index - low))


def _summary(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "max": max(samples),
        "mean": statistics.fmean(samples),
    }


class _TickRecorder:
    """Collect inner timings for the one currently executing full serving tick."""

    def __init__(self) -> None:
        self.current: dict[str, float] | None = None
        self.gc_events: list[dict[str, float | int | str]] = []

    def measure(self, name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                if self.current is not None:
                    self.current[name] = self.current.get(name, 0.0) + (
                        time.perf_counter() - started
                    ) * 1000.0

        return wrapped

    def gc_callback(self, phase: str, info: dict[str, Any]) -> None:
        now = time.perf_counter()
        if phase == "start":
            self._gc_started = now
            return
        started = getattr(self, "_gc_started", now)
        pause_ms = (now - started) * 1000.0
        event = {
            "pause_ms": pause_ms,
            "generation": int(info.get("generation", -1)),
            "collected": int(info.get("collected", 0)),
            "uncollectable": int(info.get("uncollectable", 0)),
        }
        self.gc_events.append(event)
        if self.current is not None:
            self.current["gc_pause_ms"] = self.current.get("gc_pause_ms", 0.0) + pause_ms


@contextmanager
def _instrument(recorder: _TickRecorder) -> Iterator[None]:
    """Patch only observation points; all wrapped functions keep their real bodies."""
    import src.simd_env.ego_raster_bv8 as bv8
    import web.backend.ego2s_policy as ego_policy
    import web.backend.serialize as serialize

    original_v8_scalars = bv8.v8_scalars
    original_mask = __import__("src.simd_env.ego_live_bv8", fromlist=["LiveV8Source"])
    source_type = original_mask.LiveV8Source
    gc.callbacks.append(recorder.gc_callback)
    try:
        with (
            patch.object(
                ego_policy, "forward_q", recorder.measure("forward_q_ms", ego_policy.forward_q)
            ),
            patch.object(
                bv8, "v8_scalars", recorder.measure("v8_flood_features_ms", original_v8_scalars)
            ),
            patch.object(
                source_type,
                "get_advisory_action_mask",
                recorder.measure(
                    "v8_advisory_mask_ms", source_type.get_advisory_action_mask
                ),
            ),
            patch.object(
                serialize, "build_frame", recorder.measure("snapshot_ms", serialize.build_frame)
            ),
        ):
            yield
    finally:
        gc.callbacks.remove(recorder.gc_callback)


def _advance(session: Any, count: int) -> None:
    for _ in range(count):
        before = int(session.game.frame)
        session.step()
        session.snapshot()
        measurements._assert_live_advance(session, before)


def _paced_attempt(fps: int, seed: int) -> dict[str, Any]:
    measurements._seed_everything(seed)
    measurements.set_world_seed(seed)
    session = measurements._new_session(measurements.MODE_WATCH, 12)
    measurements._assert_b8_session(session)
    if getattr(session.policy, "forward", None) != "batched":
        raise RuntimeError(
            f"diagnosis requires existing batched path, got {session.policy.forward!r}"
        )
    _advance(session, WARMUP_TICKS)
    recorder = _TickRecorder()
    rows: list[dict[str, float | int]] = []
    budget_ms = 1000.0 / fps
    original_update = session.game.update
    session.game.update = recorder.measure("game_state_update_inclusive_ms", original_update)
    try:
        with _instrument(recorder):
            for tick in range(PACED_TICKS):
                row: dict[str, float | int] = {"tick": tick}
                recorder.current = row  # GC callback attributes only this tick.
                before = int(session.game.frame)
                started = time.perf_counter()
                session.step()
                after_step = time.perf_counter()
                measurements._assert_live_advance(session, before)
                frame = session.snapshot()
                after_snapshot = time.perf_counter()
                json.dumps(frame)  # exact Hub separators/default serialization.
                finished = time.perf_counter()
                recorder.current = None
                row["session_step_ms"] = (after_step - started) * 1000.0
                row["snapshot_total_ms"] = (after_snapshot - after_step) * 1000.0
                row["json_serialization_ms"] = (finished - after_snapshot) * 1000.0
                row["full_tick_ms"] = (finished - started) * 1000.0
                row["remainder_sim_update_ms"] = max(
                    0.0,
                    float(row.get("game_state_update_inclusive_ms", 0.0))
                    - float(row.get("forward_q_ms", 0.0))
                    - float(row.get("v8_flood_features_ms", 0.0))
                    - float(row.get("v8_advisory_mask_ms", 0.0)),
                )
                rows.append(row)
                time.sleep(1.0 / fps)  # same sleep-after-work model as Hub.
    finally:
        session.game.update = original_update
    full_ticks = [float(row["full_tick_ms"]) for row in rows]
    tails = sorted(rows, key=lambda row: float(row["full_tick_ms"]), reverse=True)[:10]
    components = {
        key: _summary([float(row.get(key, 0.0)) for row in rows])
        for key in (
            "session_step_ms",
            "game_state_update_inclusive_ms",
            "forward_q_ms",
            "v8_flood_features_ms",
            "v8_advisory_mask_ms",
            "remainder_sim_update_ms",
            "snapshot_total_ms",
            "snapshot_ms",
            "json_serialization_ms",
            "gc_pause_ms",
            "full_tick_ms",
        )
    }
    return {
        "requested_fps": fps,
        "seed": seed,
        "warmup_advancing_ticks": WARMUP_TICKS,
        "paced_advancing_ticks": PACED_TICKS,
        "budget_ms": budget_ms,
        "overruns": sum(sample >= budget_ms for sample in full_ticks),
        "actual_forward_mode": session.policy.forward,
        "served": measurements._served_identity(session),
        "full_tick_ms": _summary(full_ticks),
        "components_ms": components,
        "tail_ticks": tails,
        "raw_ticks": rows,
        "gc_callbacks": recorder.gc_events,
    }


def run(output: Path) -> dict[str, Any]:
    """Run the two fixed-seed diagnostic probes and exclusively create ``output``."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    preconditions = measurements.latency_preconditions()
    torch_threads = __import__("torch").get_num_threads()
    with (
        measurements._served_b8_environment(),
        measurements._saved_global_config(),
        measurements._saved_rng_state(),
    ):
        attempts = [_paced_attempt(fps, DIAGNOSTIC_SEEDS[fps]) for fps in (24, 17)]
    payload = {
        "kind": "diagnostic_only_not_qualification",
        "purpose": "Watch12 full-tick tail decomposition after the failed run-v1 pacing attempts",
        "preconditions": preconditions,
        "cpu_threads": torch_threads,
        "protocol": {
            "mode": "watch",
            "snakes": 12,
            "warmup_advancing_ticks": WARMUP_TICKS,
            "paced_advancing_ticks": PACED_TICKS,
            "sleep_model": "full requested interval after work, matching Hub.engine_loop",
            "serialization": "json.dumps(frame) matching Hub.broadcast defaults",
            "transport": "no socket send; serialization only, so network cost is not measured",
        },
        "attempts": attempts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


class _LoopbackWebSocket:
    """Adapt a real ``websockets`` server connection to Hub's WebSocket surface."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def send_text(self, text: str) -> None:
        await self._connection.send(text)


async def _websocket_attempt() -> dict[str, Any]:
    """Measure Hub serialization/send to a client over a real loopback socket."""
    import websockets
    from web.backend.app import Hub

    received: asyncio.Queue[float] = asyncio.Queue()
    connected: asyncio.Queue[Any] = asyncio.Queue()

    async def handler(connection: Any) -> None:
        await connected.put(connection)
        await connection.wait_closed()

    async def receive_client(client: Any) -> None:
        async for _ in client:
            await received.put(time.perf_counter())

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            receiver = asyncio.create_task(receive_client(client))
            connection = await asyncio.wait_for(connected.get(), timeout=2.0)
            hub = Hub()
            hub.clients.add(_LoopbackWebSocket(connection))
            measurements._seed_everything(WEBSOCKET_SEED)
            measurements.set_world_seed(WEBSOCKET_SEED)
            session = measurements._new_session(measurements.MODE_WATCH, 12)
            measurements._assert_b8_session(session)
            if getattr(session.policy, "forward", None) != "batched":
                raise RuntimeError("loopback diagnosis requires B8's existing batched forward")
            _advance(session, WARMUP_TICKS)
            recorder = _TickRecorder()
            rows: list[dict[str, float | int]] = []
            try:
                original_update = session.game.update
                session.game.update = recorder.measure(
                    "game_state_update_inclusive_ms", original_update
                )
                try:
                    with _instrument(recorder):
                        for tick in range(PACED_TICKS):
                            row: dict[str, float | int] = {"tick": tick}
                            recorder.current = row
                            before = int(session.game.frame)
                            started = time.perf_counter()
                            session.step()
                            after_step = time.perf_counter()
                            measurements._assert_live_advance(session, before)
                            frame = session.snapshot()
                            after_snapshot = time.perf_counter()
                            await hub.broadcast(frame)
                            after_broadcast = time.perf_counter()
                            delivered = await asyncio.wait_for(received.get(), timeout=2.0)
                            recorder.current = None
                            row["session_step_ms"] = (after_step - started) * 1000.0
                            row["snapshot_total_ms"] = (after_snapshot - after_step) * 1000.0
                            row["hub_broadcast_ms"] = (after_broadcast - after_snapshot) * 1000.0
                            row["loopback_delivery_ms"] = (delivered - after_snapshot) * 1000.0
                            row["full_tick_with_broadcast_ms"] = (
                                after_broadcast - started
                            ) * 1000.0
                            row["remainder_sim_update_ms"] = max(
                                0.0,
                                float(row.get("game_state_update_inclusive_ms", 0.0))
                                - float(row.get("forward_q_ms", 0.0))
                                - float(row.get("v8_flood_features_ms", 0.0))
                                - float(row.get("v8_advisory_mask_ms", 0.0)),
                            )
                            rows.append(row)
                            await asyncio.sleep(1.0 / 24.0)
                finally:
                    session.game.update = original_update
            finally:
                receiver.cancel()
                with suppress(asyncio.CancelledError):
                    await receiver
    finally:
        server.close()
        await server.wait_closed()
    tails = sorted(rows, key=lambda row: float(row["hub_broadcast_ms"]), reverse=True)[:10]
    components = {
        key: _summary([float(row.get(key, 0.0)) for row in rows])
        for key in (
            "game_state_update_inclusive_ms",
            "forward_q_ms",
            "v8_flood_features_ms",
            "v8_advisory_mask_ms",
            "remainder_sim_update_ms",
            "snapshot_total_ms",
            "snapshot_ms",
            "hub_broadcast_ms",
            "loopback_delivery_ms",
            "gc_pause_ms",
            "full_tick_with_broadcast_ms",
        )
    }
    return {
        "requested_fps": 24,
        "seed": WEBSOCKET_SEED,
        "warmup_advancing_ticks": WARMUP_TICKS,
        "paced_advancing_ticks": PACED_TICKS,
        "actual_forward_mode": session.policy.forward,
        "served": measurements._served_identity(session),
        "components_ms": components,
        "tail_broadcast_ticks": tails,
        "raw_ticks": rows,
        "gc_callbacks": recorder.gc_events,
    }


def run_websocket(output: Path) -> dict[str, Any]:
    """Create a separate actual-loopback transport diagnostic artifact."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"loopback diagnostic output already exists: {output}")
    preconditions = measurements.latency_preconditions()
    with (
        measurements._served_b8_environment(),
        measurements._saved_global_config(),
        measurements._saved_rng_state(),
    ):
        attempt = asyncio.run(_websocket_attempt())
    payload = {
        "kind": "diagnostic_only_actual_loopback_websocket_not_qualification",
        "purpose": "Watch12 actual Hub.broadcast serialization/send timing over localhost",
        "preconditions": preconditions,
        "cpu_threads": __import__("torch").get_num_threads(),
        "transport_scope": (
            "real ephemeral localhost TCP/WebSocket server and client started by this process; "
            "Hub.broadcast includes json.dumps plus send_text; excludes remote network and "
            "client rendering"
        ),
        "attempt": attempt,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--with-websocket", action="store_true")
    parser.add_argument("--websocket-out", type=Path, default=WEBSOCKET_OUTPUT)
    args = parser.parse_args()
    if args.with_websocket:
        result = run_websocket(args.websocket_out)
        print(json.dumps({"out": str(args.websocket_out), "attempts": 1}, sort_keys=True))
    else:
        result = run(args.out)
        print(
            json.dumps({"out": str(args.out), "attempts": len(result["attempts"])}, sort_keys=True)
        )


if __name__ == "__main__":
    main()
