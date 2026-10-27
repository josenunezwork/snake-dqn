"""Real Hub/WebSocket timing arm for the first served-search measurement package.

This module is deliberately outside the serving import graph.  It starts an
unmodified :class:`web.backend.app.Hub` loop in one fresh local process and
connects an actual loopback WebSocket client from its parent.  Instrumentation
is installed only on the probe-owned instances and is removed as the child
shuts down; it never changes a released pin, endpoint, or default session.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing as mp
import socket
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Mapping

FPS = 24
WARMUP_TICKS = 200
MEASURED_TICKS = 2400
FRAME_NS = 1_000_000_000 // FPS
UPDATE_BAR_NS = 11_688_000
SCHEMA = "served-production-hub-trace/v1"


def _settings(spec: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """Validate a bounded production-arm exposure, allowing only smoke shrinkage."""
    seed = int(spec["seed"])
    warmup = int(spec.get("warmup_ticks", WARMUP_TICKS))
    measured = int(spec.get("measured_ticks", MEASURED_TICKS))
    fps = int(spec.get("fps", FPS))
    if seed < 0 or warmup < 0 or measured <= 0 or fps <= 0:
        raise ValueError("seed/warmup/measured/fps must be non-negative bounded integers")
    if warmup > WARMUP_TICKS or measured > MEASURED_TICKS or fps != FPS:
        raise ValueError("production arm only permits the frozen 24fps exposure or a smaller smoke")
    return seed, warmup, measured, fps


def _frame_id(message: Mapping[str, Any]) -> int | None:
    """Return the product frame identity without inventing a probe protocol."""
    for key in ("frame", "game_frame"):
        value = message.get(key)
        if isinstance(value, int):
            return value
    game = message.get("game")
    if isinstance(game, Mapping) and isinstance(game.get("frame"), int):
        return int(game["frame"])
    return None


def _child(connection: Any, spec: Mapping[str, Any]) -> None:
    """Own a one-shot Hub server and return bounded trace evidence over its Pipe."""
    server = None
    listener = None
    session = None
    try:
        seed, warmup, measured, fps = _settings(spec)
        from research.bl_served_search.served_tick_probe import new_probe_watch
        from research.bl_served_search.tick_trace import TraceRecorder
        from research.t1_commitment_20261015.replay import configure_cpu
        from src.data.score_store import ScoreStore
        from web.backend import app as app_module
        from web.backend import serialize

        configure_cpu()
        session, roster = new_probe_watch(seed)
        # The constructor is the product's Watch6 constructor.  Keep its explicit
        # Path-B refusal and roster check, then make the endpoint cadence explicit.
        session.set_speed(fps)
        # The production monitor writes operational telemetry.  This probe owns
        # its in-memory recorder instead, so its endpoints and monitor are
        # explicit and side-effect-free.
        session._release_monitoring = False
        # Hub dispatches ``step`` and ``snapshot`` through separate to_thread
        # calls.  They may land on different executor threads, so each has its
        # own recorder and its own thread_time_ns denominator.  Rows are joined
        # only after shutdown, never while either recorder has an active tick.
        step_recorder = TraceRecorder(capacity=warmup + measured)
        snapshot_recorder = TraceRecorder(capacity=warmup + measured)
        send_capacity = warmup + measured + 8
        send_rows: list[dict[str, Any] | None] = [None] * send_capacity
        send_count = 0
        tick_number = 0
        last_snapshot_end_ns: int | None = None
        complete = asyncio.Event()
        original_step = session.step
        original_snapshot = session.snapshot
        original_build_frame = serialize.build_frame
        # ``TraceRecorder.attach`` owns the policy/update hooks for the step
        # thread.  The following original methods are restored only while the
        # distinct snapshot thread builds its inspector DTO; otherwise an
        # inactive step recorder would correctly refuse that cross-thread span.
        snapshot_hook_originals: list[tuple[Any, str, Any]] = []
        seen_hooks: set[tuple[int, str]] = set()
        for snake in tuple(session.game.snakes):
            policy = getattr(snake, "policy", None)
            for obj, attribute in (
                (policy, "_ensure_frame"),
                (getattr(policy, "net", None), "forward"),
                (getattr(policy, "dqn", None), "forward"),
            ):
                key = (id(obj), attribute)
                original = getattr(obj, attribute, None) if obj is not None else None
                if key not in seen_hooks and callable(original):
                    seen_hooks.add(key)
                    snapshot_hook_originals.append((obj, attribute, original))

        current_target_ns = 0
        current_wake_ns = 0
        snapshot_expected = False
        completion_loop: asyncio.AbstractEventLoop | None = None
        initial_snapshot_seen = asyncio.Event()
        trace_attachment = step_recorder.attach(session)
        attachment_entered = False

        def initial_snapshot() -> Any:
            """Serve the real connect-time DTO before installing tick hooks."""
            result = original_snapshot()
            if completion_loop is not None:
                completion_loop.call_soon_threadsafe(initial_snapshot_seen.set)
            return result

        def traced_step() -> Any:
            nonlocal snapshot_expected
            wake_ns = time.perf_counter_ns()
            # Hub uses processing followed by a relative sleep, not an absolute
            # clock.  This target measures that actual cadence, including warmup.
            target_ns = wake_ns if last_snapshot_end_ns is None else last_snapshot_end_ns + FRAME_NS
            nonlocal current_target_ns, current_wake_ns
            current_target_ns, current_wake_ns = target_ns, wake_ns
            snapshot_expected = True
            step_recorder.begin_tick(tick_number, target_ns, wake_ns)
            try:
                with step_recorder.span("session_step"):
                    return original_step()
            finally:
                step_recorder.capture_session_context(session)
                step_recorder.end_tick(time.perf_counter_ns())

        def traced_build_frame(*args: Any, **kwargs: Any) -> Any:
            # The WebSocket endpoint emits one connect-time snapshot outside
            # Hub.engine_loop.  It is real product traffic but outside the
            # timed arm, so do not require an active tick for its DTO build.
            if snapshot_recorder._active < 0:
                return original_build_frame(*args, **kwargs)
            with snapshot_recorder.span("dto"):
                return original_build_frame(*args, **kwargs)

        def traced_snapshot() -> Any:
            nonlocal tick_number, last_snapshot_end_ns, snapshot_expected
            # The stream endpoint sends one initial snapshot on connect.  It is
            # product behavior but not an engine tick, so leave it outside the
            # tick-owned recorders.
            if not snapshot_expected:
                return original_snapshot()
            snapshot_recorder.begin_tick(tick_number, current_target_ns, current_wake_ns)
            restores: list[tuple[Any, str, Any]] = []
            try:
                for obj, attribute, original in snapshot_hook_originals:
                    traced = getattr(obj, attribute)
                    setattr(obj, attribute, original)
                    restores.append((obj, attribute, traced))
                with snapshot_recorder.span("snapshot"):
                    return original_snapshot()
            finally:
                for obj, attribute, traced in reversed(restores):
                    setattr(obj, attribute, traced)
                finished_ns = time.perf_counter_ns()
                snapshot_recorder.end_tick(finished_ns)
                last_snapshot_end_ns = finished_ns
                tick_number += 1
                snapshot_expected = False

        serialize.build_frame = traced_build_frame
        probe_hub = app_module.Hub()
        original_send = probe_hub._send_frame
        original_broadcast = probe_hub.broadcast
        original_apply_control = app_module._apply_control
        current_send_frame: int | None = None
        engine_error: str | None = None

        async def traced_send(ws: Any, text: str) -> str:
            # This wrapper is constructed once and runs after snapshot work.
            # It records the actual Hub send boundary without disk or log I/O.
            started_ns = time.perf_counter_ns()
            status = await original_send(ws, text)
            ended_ns = time.perf_counter_ns()
            nonlocal send_count
            if send_count >= send_capacity:
                raise RuntimeError("bounded frame-send recorder exhausted")
            send_rows[send_count] = {
                "frame": current_send_frame,
                "start_ns": started_ns,
                "end_ns": ended_ns,
                "status": status,
            }
            send_count += 1
            return status

        async def traced_broadcast(message: dict) -> None:
            # Extract identity before the production broadcaster serializes once;
            # do not parse its text again or add parser/GC work to the tick.
            nonlocal current_send_frame, engine_error
            current_send_frame = _frame_id(message)
            await original_broadcast(message)
            if tick_number == warmup + measured:
                # Stop admission immediately after the last measured frame has
                # been handed to the actual Hub broadcaster.  Uvicorn observes
                # ``should_exit`` on a coarser cadence, so waiting for the
                # parent receipt acknowledgement alone could admit one extra
                # 24fps step before server shutdown.  Cancelling this exact
                # probe-owned engine task lands at Hub's next sleep, after the
                # completed send; the server remains up for the acknowledgement.
                engine_task = probe_hub.engine_task
                if engine_task is None or engine_task is not asyncio.current_task():
                    raise RuntimeError("final broadcast did not run on the owned Hub engine task")
                engine_task.cancel()
            if message.get("type") == "error":
                engine_error = str(message.get("message"))
                if completion_loop is not None:
                    completion_loop.call_soon(complete.set)

        def probe_apply_control(sess: Any, message: object, conn_id: object = None) -> Any:
            if isinstance(message, Mapping) and message.get("action") == "probe_complete":
                if completion_loop is None:
                    raise RuntimeError("probe completion arrived before server loop startup")
                completion_loop.call_soon_threadsafe(complete.set)
                return None
            return original_apply_control(sess, message, conn_id)

        probe_hub._send_frame = traced_send
        probe_hub.broadcast = traced_broadcast
        app_module._apply_control = probe_apply_control
        app_module.hub = probe_hub
        score_dir = tempfile.TemporaryDirectory(prefix="served-production-probe-")
        score_path = Path(score_dir.name) / "scores.db"

        @asynccontextmanager
        async def probe_lifespan(_: Any):
            probe_hub.store = ScoreStore(str(score_path))
            probe_hub.session = session

            async def start_engine_after_connect_snapshot() -> None:
                nonlocal attachment_entered
                await initial_snapshot_seen.wait()
                trace_attachment.__enter__()
                attachment_entered = True
                # The production endpoint has sent its ordinary initial frame.
                # From this point onward every frame comes from Hub.engine_loop.
                session.step = traced_step
                session.snapshot = traced_snapshot
                probe_hub.engine_task = asyncio.create_task(probe_hub.engine_loop())

            starter = asyncio.create_task(start_engine_after_connect_snapshot())
            try:
                yield
            finally:
                starter.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await starter
                if probe_hub.engine_task is not None:
                    probe_hub.engine_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await probe_hub.engine_task
                if probe_hub.store is not None:
                    probe_hub.store.close()
                app_module._apply_control = original_apply_control
                if attachment_entered:
                    trace_attachment.__exit__(None, None, None)
                await asyncio.to_thread(session.close, True)
                score_dir.cleanup()

        app_module.app.router.lifespan_context = probe_lifespan
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(5)
        port = int(listener.getsockname()[1])
        connection.send({"kind": "ready", "port": port})

        async def run_server() -> None:
            import uvicorn

            nonlocal server, completion_loop
            completion_loop = asyncio.get_running_loop()
            # ``ws_stream`` sends this frame itself before the actual Hub loop
            # starts; the start gate prevents a non-tick inspector from touching
            # tick-only policy wrappers.
            session.snapshot = initial_snapshot
            config = uvicorn.Config(
                app_module.app,
                host="127.0.0.1",
                port=port,
                log_level="error",
                timeout_graceful_shutdown=1,
            )
            server = uvicorn.Server(config)

            async def stop_when_complete() -> None:
                await complete.wait()
                server.should_exit = True

            stopper = asyncio.create_task(stop_when_complete())
            try:
                await server.serve(sockets=[listener])
            finally:
                stopper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stopper

        try:
            asyncio.run(run_server())
        finally:
            serialize.build_frame = original_build_frame
        step_rows = step_recorder.rows()[warmup:]
        snapshot_rows = snapshot_recorder.rows()[warmup:]
        if len(step_rows) != measured or len(snapshot_rows) != measured:
            raise RuntimeError("production recorder lost a warmup or measured tick")
        trace_rows: list[dict[str, Any]] = []
        for index, (step_row, snapshot_row) in enumerate(zip(step_rows, snapshot_rows)):
            if step_row["tick"] != snapshot_row["tick"] or step_row["tick"] != warmup + index:
                raise RuntimeError("step/snapshot recorder identity mismatch")
            merged = dict(step_row)
            merged["tick"] = index
            merged["snapshot_thread_id"] = snapshot_row["thread_id"]
            merged["snapshot_start_ns"] = snapshot_row["start_ns"]
            merged["snapshot_end_ns"] = snapshot_row["end_ns"]
            merged["snapshot_start_cpu_ns"] = snapshot_row["start_cpu_ns"]
            merged["snapshot_end_cpu_ns"] = snapshot_row["end_cpu_ns"]
            merged["spans"] = dict(step_row["spans"])
            merged["spans"]["snapshot"] = snapshot_row["spans"]["snapshot"]
            merged["spans"]["dto"] = snapshot_row["spans"]["dto"]
            trace_rows.append(merged)
        connection.send(
            {
                "kind": "result",
                "result": {
                    "schema": SCHEMA,
                    "arm": str(spec.get("arm", "production")),
                    "seed": seed,
                    "fps": fps,
                    "warmup_ticks": warmup,
                    "measured_ticks": measured,
                    "clock_kind": "hub_relative_sleep_not_absolute",
                    "absolute_overrun_applicable": False,
                    "update_p95_limit_ns": UPDATE_BAR_NS,
                    "roster": dict(roster),
                    "trace": trace_rows,
                    "frame_send": [row for row in send_rows[:send_count] if row is not None],
                    "tick_thread": "asyncio.to_thread(GameSession.step/snapshot)",
                    "score_db": "temporary-child-owned",
                    "endpoints": {"stream": "/ws/stream", "monitoring": "disabled"},
                    "engine_error": engine_error,
                },
            }
        )
    except BaseException as exc:
        with contextlib.suppress(Exception):
            connection.send({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if listener is not None:
            listener.close()
        connection.close()


async def _collect_client(
    port: int, wanted_frames: int, timeout_seconds: float
) -> list[dict[str, Any]]:
    """Collect same-Mac WebSocket receipt times until the owned server closes."""
    import websockets

    uri = f"ws://127.0.0.1:{port}/ws/stream"
    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(
                uri, open_timeout=min(2.0, max(0.1, deadline - time.monotonic()))
            ) as ws:
                rows: list[dict[str, Any]] = []
                first_frame = True
                received_engine_frames = 0
                acknowledgement_sent = False
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("WebSocket client timed out before server completion")
                    try:
                        text = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except websockets.ConnectionClosed:
                        break
                    received_ns = time.perf_counter_ns()
                    payload = json.loads(text)
                    if not isinstance(payload, Mapping):
                        continue
                    frame = _frame_id(payload)
                    if frame is not None:
                        rows.append(
                            {
                                "frame": frame,
                                "receive_ns": received_ns,
                                "source": "connect_snapshot" if first_frame else "hub_broadcast",
                            }
                        )
                        if not first_frame:
                            received_engine_frames += 1
                        first_frame = False
                    if len(rows) > wanted_frames + 9:
                        raise RuntimeError(
                            "client received more frames than bounded probe exposure"
                        )
                    if received_engine_frames >= wanted_frames and not acknowledgement_sent:
                        await ws.send(json.dumps({"type": "control", "action": "probe_complete"}))
                        acknowledgement_sent = True
                return rows
        except (OSError, asyncio.TimeoutError) as exc:
            last_error = exc
            await asyncio.sleep(0.02)
    raise RuntimeError(f"production probe could not connect to owned server: {last_error}")


def run_production_arm(spec: dict, timeout_seconds: float = 900) -> dict[str, Any]:
    """Run one production-equivalent Hub/WebSocket arm in an owned child process.

    The parent performs no game work.  It only receives actual loopback WebSocket
    frames and merges their timestamps with the child-owned game-thread trace.
    Child startup, transport, or cleanup failures refuse the entire arm.
    """
    _, warmup, measured, _ = _settings(spec)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(child, dict(spec)), daemon=False)
    process.start()
    child.close()
    started = time.monotonic()
    try:
        if not parent.poll(min(60.0, timeout_seconds)):
            raise TimeoutError("production server did not report a loopback port")
        ready = parent.recv()
        if ready.get("kind") != "ready":
            raise RuntimeError(f"production server startup failed: {ready.get('error')}")
        remaining = timeout_seconds - (time.monotonic() - started)
        client_rows = asyncio.run(_collect_client(int(ready["port"]), warmup + measured, remaining))
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0 or not parent.poll(remaining):
            raise TimeoutError("production server did not return a bounded result")
        message = parent.recv()
        if message.get("kind") != "result":
            raise RuntimeError(f"production server failed: {message.get('error')}")
        result = dict(message["result"])
        result["client_receive"] = client_rows
        result["client_transport"] = "parent-loopback-websocket-perf_counter_ns"
        if result.get("engine_error"):
            raise RuntimeError(f"Hub.engine_loop failed: {result['engine_error']}")
        if len(result["trace"]) != measured:
            raise RuntimeError("production trace incomplete")
        return result
    finally:
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()  # Exact probe-owned child only.
            process.join(timeout=10)
        if process.exitcode not in (0, None):
            raise RuntimeError(f"production server child exit={process.exitcode}")
