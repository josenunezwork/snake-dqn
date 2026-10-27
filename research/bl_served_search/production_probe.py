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
import sys
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
FINAL_ACK_TIMEOUT_SECONDS = 10.0
CLIENT_INACTIVITY_TIMEOUT_SECONDS = 10.0
STARTUP_TIMEOUT_SECONDS = 20.0
RESULT_TIMEOUT_SECONDS = 20.0


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


def _merge_span(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Combine same-named independent-thread span totals into one trace row."""
    result = dict(left)
    for name in (
        "calls",
        "wall_ns",
        "wall_exclusive_ns",
        "cpu_ns",
        "cpu_exclusive_ns",
    ):
        result[name] = int(left.get(name, 0)) + int(right.get(name, 0))
    starts = [
        int(value)
        for value in (left.get("first_wall_start_ns"), right.get("first_wall_start_ns"))
        if value
    ]
    ends = [
        int(value)
        for value in (left.get("last_wall_end_ns"), right.get("last_wall_end_ns"))
        if value
    ]
    result["first_wall_start_ns"] = min(starts) if starts else 0
    result["last_wall_end_ns"] = max(ends) if ends else 0
    # Durations add; thread CPU timestamps do not share a clock domain. Retain
    # the original intervals as components instead of inventing a combined one.
    result.pop("first_cpu_start_ns", None)
    result.pop("last_cpu_end_ns", None)
    result["cpu_clock_kind"] = "aggregate_independent_thread_durations"
    result["components"] = {"step": dict(left), "snapshot": dict(right)}
    return result


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
        current_target_ns = 0
        current_wake_ns = 0
        snapshot_expected = False
        completion_loop: asyncio.AbstractEventLoop | None = None
        client_ready = asyncio.Event()
        trace_attachment = step_recorder.attach(session)
        snapshot_telemetry_attachment = snapshot_recorder.attach_telemetry(session)
        attachment_entered = False

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
            try:
                with snapshot_recorder.span("snapshot"):
                    return original_snapshot()
            finally:
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

                # A dropped final frame must not leave the owned server waiting
                # for an acknowledgement until the panel-wide timeout.  The
                # parent rejects the partial receipt set, while this timer gives
                # the child a bounded, clean shutdown path.
                async def finish_without_ack() -> None:
                    await asyncio.sleep(FINAL_ACK_TIMEOUT_SECONDS)
                    complete.set()

                asyncio.create_task(finish_without_ack())
            if message.get("type") == "error":
                engine_error = str(message.get("message"))
                if completion_loop is not None:
                    completion_loop.call_soon(complete.set)

        def probe_apply_control(sess: Any, message: object, conn_id: object = None) -> Any:
            if isinstance(message, Mapping) and message.get("action") == "probe_ready":
                if completion_loop is None:
                    raise RuntimeError("probe readiness arrived before server loop startup")
                completion_loop.call_soon_threadsafe(client_ready.set)
                return None
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
                # The loop begins only after the actual client has received the
                # ordinary connect-time DTO and sent an explicit control reply.
                # This prevents tick zero from racing client registration or the
                # endpoint's initial send.
                await client_ready.wait()
                trace_attachment.__enter__()
                snapshot_telemetry_attachment.__enter__()
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
                    snapshot_telemetry_attachment.__exit__(None, None, None)
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
            merged["spans"]["ipc_lock"] = _merge_span(
                step_row["spans"]["ipc_lock"], snapshot_row["spans"]["ipc_lock"]
            )
            for field in ("gc_start", "gc_stop", "gc_overlap_ns"):
                merged[field] = [
                    int(step) + int(snapshot)
                    for step, snapshot in zip(step_row[field], snapshot_row[field])
                ]
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
    """Collect one same-Mac WebSocket receipt stream from the owned server.

    Connection retries cover startup only.  Once connected, an early close,
    receive timeout, or malformed completion is evidence of an invalid probe;
    reconnecting would create a second product snapshot mid-stream.
    """
    import websockets

    uri = f"ws://127.0.0.1:{port}/ws/stream"
    deadline = time.monotonic() + timeout_seconds
    connect_deadline = min(deadline, time.monotonic() + STARTUP_TIMEOUT_SECONDS)
    last_error: BaseException | None = None
    ws = None
    while time.monotonic() < connect_deadline:
        try:
            ws = await websockets.connect(
                uri, open_timeout=min(2.0, max(0.001, connect_deadline - time.monotonic()))
            )
            break
        except (OSError, asyncio.TimeoutError) as exc:
            last_error = exc
            await asyncio.sleep(0.02)
    if ws is None:
        raise RuntimeError(f"production probe could not connect to owned server: {last_error}")
    try:
        rows: list[dict[str, Any]] = []
        first_frame = True
        received_engine_frames = 0
        acknowledgement_sent = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("WebSocket client timed out before server completion")
            try:
                text = await asyncio.wait_for(
                    ws.recv(), timeout=min(CLIENT_INACTIVITY_TIMEOUT_SECONDS, remaining)
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError("WebSocket client timed out before server completion") from exc
            except websockets.ConnectionClosed as exc:
                if received_engine_frames != wanted_frames:
                    raise RuntimeError(
                        "production WebSocket closed before all engine frames"
                    ) from exc
                break
            received_ns = time.perf_counter_ns()
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                continue
            frame = _frame_id(payload)
            if frame is None:
                continue
            rows.append(
                {
                    "frame": frame,
                    "receive_ns": received_ns,
                    "source": "connect_snapshot" if first_frame else "hub_broadcast",
                }
            )
            if first_frame:
                await asyncio.wait_for(
                    ws.send(json.dumps({"type": "control", "action": "probe_ready"})),
                    timeout=min(CLIENT_INACTIVITY_TIMEOUT_SECONDS, remaining),
                )
            else:
                received_engine_frames += 1
            first_frame = False
            if len(rows) > wanted_frames + 1:
                raise RuntimeError("client received more frames than bounded probe exposure")
            if received_engine_frames == wanted_frames and not acknowledgement_sent:
                await asyncio.wait_for(
                    ws.send(json.dumps({"type": "control", "action": "probe_complete"})),
                    timeout=min(CLIENT_INACTIVITY_TIMEOUT_SECONDS, remaining),
                )
                acknowledgement_sent = True
        if not acknowledgement_sent or len(rows) != wanted_frames + 1:
            raise RuntimeError("production client receipt count incomplete")
        return rows
    finally:
        active_exception = sys.exc_info()[0] is not None
        try:
            await asyncio.wait_for(ws.close(), timeout=CLIENT_INACTIVITY_TIMEOUT_SECONDS)
        except Exception:
            if not active_exception:
                raise


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
        if not parent.poll(min(STARTUP_TIMEOUT_SECONDS, timeout_seconds)):
            raise TimeoutError("production server did not report a loopback port")
        ready = parent.recv()
        if ready.get("kind") != "ready":
            raise RuntimeError(f"production server startup failed: {ready.get('error')}")
        remaining = timeout_seconds - (time.monotonic() - started)
        client_rows = asyncio.run(_collect_client(int(ready["port"]), warmup + measured, remaining))
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0 or not parent.poll(min(RESULT_TIMEOUT_SECONDS, remaining)):
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
        active_exception = sys.exc_info()[0] is not None
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()  # Exact probe-owned child only.
            process.join(timeout=10)
        if not active_exception and process.exitcode not in (0, None):
            raise RuntimeError(f"production server child exit={process.exitcode}")
