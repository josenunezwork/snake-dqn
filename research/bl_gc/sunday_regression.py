"""Round-25 d.4: real web/serve.py, real WebSocket clients, one separate look.

Operator-only. No server or timer starts on import, --help, or reducer tests.
Watch count is a qualification-local constructor override, as in T15 episodes;
the actual serve.py launcher, app lifespan, engine loop and wire format are used.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from research.bl_gc import common
from research.bl_gc.probe import _observe, _timing_admission
from research.bl_served_search.exports import verify_export

SCHEMA = "bl-gc-sunday-regression/v1"
WATCH_SIZES = (6, 8, 12)
SEED = 2026102305
GAP_MS = 41.667
CEILINGS = {6: 24.0, 8: 12.0, 12: 12.0}


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _health(port: int, process: subprocess.Popen[Any]) -> None:
    until = time.monotonic() + 60
    while time.monotonic() < until:
        if process.poll() is not None:
            raise RuntimeError("server exited before health")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as reply:
                if reply.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("health timeout")


def _validate_state(state: dict, policy: str, count: int, checkpoint: str | None) -> None:
    expected = "champion_a5_freespace_20260621.pth" if checkpoint else "b8_s3_t7_20261016.pth"
    ceiling = 120.0 if checkpoint else CEILINGS[count]
    if (
        state.get("error")
        or state.get("mode") != "watch"
        or state.get("gc_policy") != policy
        or state.get("num_snakes") != count
        or state.get("checkpoint") != expected
        or state.get("speed_ceiling") != ceiling
        or state.get("speed") != ceiling
    ):
        raise RuntimeError(f"unexpected acknowledged server state: {state}")
    if not checkpoint and state.get("watch_roster") != "mixed":
        raise RuntimeError("Sunday requires the released mixed Watch roster")


async def _client(port: int, soak: int, policy: str, count: int, checkpoint: str | None) -> dict:
    import websockets

    controls: list[dict] = []
    frames: list[dict] = []
    start_wall = 0.0

    async def receive(ws: Any, *, retain: bool = False) -> dict:
        message = json.loads(await asyncio.wait_for(ws.recv(), 15))
        if message.get("type") == "error":
            raise RuntimeError(f"server error: {message}")
        if retain and message.get("type") == "frame":
            # Keep the delivered frame identity/timestamps, not thousands of
            # inspector DTO graphs in the client process during a memory soak.
            frames.append(
                {
                    "received_at": time.time(),
                    "received_ns": time.monotonic_ns(),
                    "frame": message["frame"],
                    "paused": message["paused"],
                    "checkpoint": message["checkpoint_name"],
                }
            )
        return message

    async def frame(ws: Any) -> dict:
        while True:
            message = await receive(ws)
            if message.get("type") == "frame":
                return message

    async def control(ws: Any, action: str, value: Any = None) -> dict:
        await ws.send(json.dumps({"type": "control", "action": action, "value": value}))
        while True:
            message = await receive(ws)
            if message.get("type") == "ack":
                state = message["state"]
                if state.get("error"):
                    raise RuntimeError(f"control {action}: {state['error']}")
                controls.append({"action": action, "state": state})
                return state

    uri = f"ws://127.0.0.1:{port}/ws/stream"
    async with websockets.connect(uri, open_timeout=15, max_size=16 * 1024 * 1024) as ws:
        await frame(ws)
        state = await control(ws, "set_speed", 120)
        _validate_state(state, policy, count, checkpoint)
        state = await control(ws, "pause")
        if state["playing"]:
            raise RuntimeError("pause acknowledgement still playing")
        paused = [await frame(ws) for _ in range(3)]
        if not paused[-1]["paused"] or paused[-1]["frame"] != paused[-2]["frame"]:
            raise RuntimeError("pause did not freeze the game")
        state = await control(ws, "play")
        if not state["playing"]:
            raise RuntimeError("resume acknowledgement still paused")
        resumed = [await frame(ws) for _ in range(3)]
        if resumed[-1]["frame"] <= resumed[0]["frame"]:
            raise RuntimeError("resume did not advance frames")
        await control(ws, "reset")
        # The next broadcast is generated from the actual reset world. Old
        # in-flight frames may arrive before it; accept no invented reset ack.
        until = time.monotonic() + 5
        while True:
            fresh = await frame(ws)
            if fresh["frame"] <= 2:
                break
            if time.monotonic() >= until:
                raise RuntimeError("reset frame never arrived")
    async with websockets.connect(uri, open_timeout=15, max_size=16 * 1024 * 1024) as ws:
        first = await frame(ws)
        if first["checkpoint_name"] != fresh["checkpoint_name"]:
            raise RuntimeError("reconnect changed checkpoint")
        start_wall = time.time()
        start_ns = time.monotonic_ns()
        deadline = time.monotonic() + soak
        while time.monotonic() < deadline:
            await receive(ws, retain=True)
        end_ns = time.monotonic_ns()
    return {
        "control_states": controls,
        "reconnected": True,
        "reset_observed": True,
        "soak_start": start_wall,
        "soak_elapsed_seconds": (end_ns - start_ns) / 1e9,
        "frames": frames,
    }


def _records(log: Path) -> list[dict]:
    records = []
    for line in log.read_text(errors="replace").splitlines():
        if "b8-release-monitor:" in line:
            records.append(json.loads(line.split("b8-release-monitor:", 1)[1].strip()))
        elif line.startswith("bl-gc-sunday-attestation: "):
            records.append(
                {"event": "qualification_attestation", **json.loads(line.split(": ", 1)[1])}
            )
    return records


def _gaps(row: dict) -> list[dict]:
    frames = row["client"]["frames"]
    return [
        {
            "start": a["received_at"],
            "end": b["received_at"],
            "gap_ms": (b["received_ns"] - a["received_ns"]) / 1e6,
            "end_frame": b["frame"],
        }
        for a, b in zip(frames, frames[1:])
    ]


def _recompute_matches(row: dict) -> list[dict]:
    """Drain at tick j covers snapshot/broadcast after tick j-1 through step j."""
    ticks = row["watch_ticks"]
    gaps = _gaps(row) if "frames" in row["client"] else row["client"]["gaps"]
    intervals = [
        (float(previous["timestamp"]), float(current["timestamp"]), current)
        for previous, current in zip(ticks, ticks[1:])
    ]
    return [
        gap
        for gap in gaps
        if gap["gap_ms"] >= GAP_MS
        and any(
            begin <= gap["end"]
            and end >= gap["start"]
            and float(tick.get("gc_gen2_pause_ms", 0.0)) > 0
            for begin, end, tick in intervals
        )
    ]


def _case(policy: str, count: int, soak: int, out: Path, checkpoint: str | None = None) -> dict:
    port = _port()
    stem = f"watch{count}-{policy}" + (f"-{checkpoint}" if checkpoint else "")
    log, scores = out / f"{stem}.stderr.log", out / f"{stem}.scores.sqlite"
    env = {key: value for key, value in os.environ.items() if not key.startswith("SNAKE_SERVE_")}
    env.update(
        SNAKE_SERVE_GC_POLICY=policy,
        SNAKE_SERVE_PATHB="0",
        SNAKE_SERVE_WATCH_ROSTER="mixed",
        SNAKE_SCORES_DB=str(scores),
        SNAKE_DQN_DEVICE="cpu",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    if checkpoint:
        env["SNAKE_SERVE_CHECKPOINT"] = checkpoint
    with log.open("x") as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "research.bl_gc.sunday_server",
                "--count",
                str(count),
                "--seed",
                str(SEED),
                "--port",
                str(port),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )
    try:
        _health(port, process)
        client = asyncio.run(
            asyncio.wait_for(_client(port, soak, policy, count, checkpoint), soak + 90)
        )
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(10)
    records = _records(log)
    attests = [r for r in records if r["event"] == "qualification_attestation"]
    if len(attests) != 1 or attests[0]["torch_num_threads"] != 1 or attests[0]["pathb"] != "0":
        raise RuntimeError("missing serving process d.0 attestation")
    policies = [r for r in records if r["event"] == "gc_policy"]
    resolutions = [r for r in records if r["event"] == "resolution"]
    if len(policies) != 1 or policies[0]["policy"] != policy:
        raise RuntimeError("missing or wrong production startup GC receipt")
    if not resolutions or any(r.get("fallback_reason") for r in resolutions):
        raise RuntimeError("missing resolution or serving fallback")
    expected_name = checkpoint or "b8-s3"
    if resolutions[-1]["name"] != expected_name:
        raise RuntimeError("unexpected served checkpoint resolution")
    return {
        "policy": policy,
        "watch_size": count,
        "checkpoint_env": checkpoint,
        "server_pid": process.pid,
        "server_exit": process.returncode,
        "verified_stop": process.returncode in (0, -15),
        "attestation": attests[0],
        "stderr": str(log),
        "stderr_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "client": client,
        "gc_policy_receipt": policies[0],
        "resolutions": resolutions,
        "watch_ticks": [r for r in records if r["event"] == "watch_tick"],
    }


def run(out: Path, soak: int) -> dict:
    from research.bl_k1.m1 import HostMonitor

    if soak != 600:
        raise ValueError("Sunday requires 600-second soaks")
    if out.exists():
        raise FileExistsError(f"fresh Sunday output required: {out}")
    idle = _timing_admission()
    out.mkdir(parents=True)
    monitor = HostMonitor(_observe)
    rows, rollbacks = [], []
    reason = None
    monitor.thread.start()
    try:
        for checkpoint, count in [(None, n) for n in WATCH_SIZES] + [("champion", 6)]:
            for policy in ("default", "freeze_startup"):
                if monitor.error:
                    raise RuntimeError(monitor.error)
                row = _case(policy, count, soak, out, checkpoint)
                common.write_result(out / f"case-{count}-{policy}-{checkpoint or 'b8'}.json", row)
                (rollbacks if checkpoint else rows).append(row)
        reason = monitor.error
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        monitor.close()
    return {
        "schema": SCHEMA + "/raw",
        "seed": SEED,
        "soak_seconds": soak,
        "rows": rows,
        "checkpoint_rollback_restarts": rollbacks,
        "idle": idle,
        "host_observations": monitor.rows,
        "source_hashes": common._source_hashes(),
        "judgment_sha256": common.JUDGMENT_SHA256,
        "git_commit": common._git_commit(),
        "status": "INCOMPLETE" if reason else "COMPLETE_ONE_LOOK_PENDING",
        "reason": reason,
    }


def check(raw: dict) -> dict:
    """Check exact matrix, real control proof and paired client-visible GC gaps."""
    if (
        raw.get("schema") != SCHEMA + "/raw"
        or raw.get("status") != "COMPLETE_ONE_LOOK_PENDING"
        or raw.get("seed") != SEED
        or raw.get("soak_seconds") != 600
        or raw.get("judgment_sha256") != common.JUDGMENT_SHA256
        or not raw.get("mac_lease")
        or not raw.get("source_hashes")
        or not raw.get("host_observations")
    ):
        raise ValueError("incomplete Sunday attestation")
    rows, rollbacks = raw["rows"], raw["checkpoint_rollback_restarts"]
    expected = {(p, n) for p in ("default", "freeze_startup") for n in WATCH_SIZES}
    if len(rows) != 6 or {(r["policy"], r["watch_size"]) for r in rows} != expected:
        raise ValueError("incomplete Sunday Watch matrix")
    if len(rollbacks) != 2 or {r["policy"] for r in rollbacks} != {"default", "freeze_startup"}:
        raise ValueError("missing GC/default or champion rollback restart")
    if any(row.get("checkpoint_env") is not None for row in rows) or any(
        row.get("checkpoint_env") != "champion" or row.get("watch_size") != 6 for row in rollbacks
    ):
        raise ValueError("foreign checkpoint or rollback matrix")
    for row in rows + rollbacks:
        if (
            row["gc_policy_receipt"]["policy"] != row["policy"]
            or row["attestation"]["source_hashes"] != raw["source_hashes"]
        ):
            raise ValueError("Sunday source/policy binding mismatch")
        client = row["client"]
        if (
            not row["verified_stop"]
            or client["soak_elapsed_seconds"] < 600
            or not client["reconnected"]
            or not client["reset_observed"]
            or len(client["frames"]) < 2
            or not row["resolutions"]
        ):
            raise ValueError("incomplete Sunday client/stop proof")
        controls = client["control_states"]
        if [c["action"] for c in controls] != ["set_speed", "pause", "play", "reset"]:
            raise ValueError("missing real control acknowledgements")
        for control in controls:
            _validate_state(
                control["state"], row["policy"], row["watch_size"], row["checkpoint_env"]
            )
        if controls[1]["state"]["playing"] or not controls[2]["state"]["playing"]:
            raise ValueError("pause/resume acknowledgements invalid")
        for frame in client["frames"]:
            if not all(
                math.isfinite(frame[key]) for key in ("received_at", "received_ns", "frame")
            ):
                raise ValueError("invalid client clock")
        if row["checkpoint_env"] is None and not row["watch_ticks"]:
            raise ValueError("missing B8 watch telemetry")
    comparisons = []
    for size in WATCH_SIZES:
        default = next(r for r in rows if r["watch_size"] == size and r["policy"] == "default")
        frozen = next(
            r for r in rows if r["watch_size"] == size and r["policy"] == "freeze_startup"
        )
        baseline = {g["end_frame"]: g["gap_ms"] for g in _gaps(default)}
        matched = _recompute_matches(frozen)
        new = [g for g in matched if g["gap_ms"] > baseline.get(g["end_frame"], 0.0)]
        comparisons.append(
            {
                "watch_size": size,
                "default_gc_gaps": len(_recompute_matches(default)),
                "freeze_gc_gaps": len(matched),
                "new_gc_gaps": new,
            }
        )
    return {
        "schema": SCHEMA + "/check",
        "verdict": "FAIL" if any(c["new_gc_gaps"] for c in comparisons) else "PASS",
        "comparisons": comparisons,
        "rollback_restarts_verified": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--check", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.run == bool(args.check):
        parser.error("use exactly one of --run or --check")
    if args.run:
        from research.mac_lease import held

        with held("bl-gc-sunday", "Round-25 Sunday serving regression", seconds=6600) as lease:
            result = run(args.out, 600)
            result["mac_lease"] = lease.receipt
            common.write_result(args.out / "SUNDAY_RAW.json", result)
    else:
        verify_export(args.check)
        result = check(json.loads(args.check.read_text()))
        common.write_result(args.out, result)
    print(
        json.dumps({"out": str(args.out), "verdict": result.get("verdict", result.get("status"))})
    )
    return (
        0
        if result.get("verdict", result.get("status")) in ("PASS", "COMPLETE_ONE_LOOK_PENDING")
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
